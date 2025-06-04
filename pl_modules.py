import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import MMD

#------------------------------MLPLightningModule--------------------------------------------#
class MLPLightning(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss() if model.classification else nn.MSELoss()
        self.save_hyperparameters(ignore=['model'])
        self.monitor = 'validation_loss'
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.predict_with_logits(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.predict_with_logits(x)
        loss = self.loss_fn(logits, y)

        if self.model.classification:
            preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            y_true, y_pred = y.cpu(), preds.cpu()
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            self.log_dict({
                self.monitor: loss,
                "val_acc": acc,
                "val_f1": f1,
                "val_precision": prec,
                "val_recall": rec
            }, prog_bar=True)
        else:
            rmse = torch.sqrt(loss)
            self.log_dict({
                self.monitor: loss,
                "val_rmse": rmse
            }, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.predict_with_logits(x)
        loss = self.loss_fn(logits, y)

        if self.model.classification:
            preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            self.test_preds.append(preds.cpu().numpy())
            self.test_targets.append(y.cpu().numpy())
        else:
            self.test_preds.append(logits.cpu().numpy())
            self.test_targets.append(y.cpu().numpy())

        
        self.log("test_loss", loss)

    def on_test_end(self):
        preds = np.concatenate(self.test_preds)
        targets = np.concatenate(self.test_targets)

        if self.model.classification:
            acc = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average='macro')
            prec = precision_score(targets, preds, average='macro', zero_division=0)
            rec = recall_score(targets, preds, average='macro', zero_division=0)

            self.test_metrics = {
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec
            }

            # Log to TensorBoard
            self.logger.experiment.add_scalar("test/test_acc", acc, self.global_step)
            self.logger.experiment.add_scalar("test/test_f1", f1, self.global_step)
            self.logger.experiment.add_scalar("test/test_precision", prec, self.global_step)
            self.logger.experiment.add_scalar("test/test_recall", rec, self.global_step)


        else:
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            self.test_metrics = {
                "rmse": rmse
            }

            # Log to TensorBoard
            self.logger.experiment.add_scalar("test/test_rmse", rmse, self.global_step)



        print("Test Metrics:", self.test_metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def checkpoint(self):
        return ModelCheckpoint(save_top_k=1,mode='min',save_last=False,monitor=self.monitor)
    
#------------------------------MLPLightningModule--------------------------------------------#    




#------------------------------CausalNFLightningModule--------------------------------------------#  

class CausalNF(pl.LightningModule):
    def __init__(self, flow, lr ):
        super(CausalNF,self).__init__()
        self.monitor = 'validation_loss'
        self.save_hyperparameters(ignore=['flow'])
        self.lr = lr
        #self.feature = feature_type_num
        self.flow=flow

    def training_step(self,batch,batch_idx):
        model=self.flow()   
        x, y = batch
        x = x.float().to(self.device)
        #x = x + self.feature.to(self.device) * torch.rand(x.shape, device=self.device) #*0.01
        y = y.float().to(self.device) 
        u_sample = model.base.sample((x.shape[0],)).to(self.device)
        x_sample = model.transform(u_sample)
        loss = MMD(x_sample,x,lengthscale=2.0)
        self.log('training_loss',loss)  # #
        #if self.current_epoch%20==0 and batch_idx==0:
        #    print(f'training---current_epoch:{self.current_epoch} batch_idx:{batch_idx} loss:{loss}')
        return loss


        
    def validation_step(self,batch,batch_idx):
        model=self.flow()
        x, y = batch
        x = x.float().to(self.device)
        #x = x + self.feature.to(self.device) * torch.rand(x.shape, device=self.device) #* 0.01
        y = y.float().to(self.device) 

        u_sample = model.base.sample((x.shape[0],)).to(self.device)
        x_sample = model.transform(u_sample)
        loss = MMD(x_sample,x,lengthscale=2.0)  # #
        self.log(self.monitor,loss)
        #if self.current_epoch%100==0 and batch_idx%5 ==0:
        #    print(f'current_epoch:{self.current_epoch} batch_idx:{batch_idx} loss:{loss}')
        return loss
    
    def test_step(self,batch,batch_idx):
        model=self.flow()
        x, y = batch
        x = x.float().to(self.device)
        y = y.float().to(self.device)
        u_sample = model.base.sample((x.shape[0],)).to(self.device)
        x_sample = model.transform(u_sample)
        loss = MMD(x_sample,x,lengthscale=2.0)  # #
        self.log('test_loss',loss,prog_bar=True, on_epoch=True, logger=True)
        return loss

    def checkpoint(self):
        return ModelCheckpoint(save_top_k=1,mode='min',save_last=False,monitor=self.monitor)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.flow.parameters(),lr=self.lr)
        return [opt]
    
#------------------------------CausalNFLightningModule--------------------------------------------#  
