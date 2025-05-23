import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint

activation_functions = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(),
                        'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation = 'relu',classification = True,bias = True):
        super().__init__()
        self.name = 'ArtificialNeuralNetwork'
        self.abbrv = 'ann'
        self.classification = classification
        self.bias = bias
        
        # Construct layers
        model_layers = []
        previous_layer = input_dim
        for layer in hidden_layers:
            model_layers.append(nn.Linear(previous_layer, layer, bias=self.bias))
            model_layers.append(activation_functions[activation])
            previous_layer = layer
        n_class = 2 if classification else 1    
        model_layers.append(nn.Linear(previous_layer, n_class))
        self.network = nn.Sequential(*model_layers)
    
    def predict_layer(self, x, hidden_layer_idx=0, post_act=True):
        if hidden_layer_idx >= len(self.network) // 2:
            raise ValueError(f'The model has only {len(self.network) // 2} hidden layers, but hidden layer {hidden_layer_idx} was requested (indexing starts at 0).')
        
        network_idx = 2 * hidden_layer_idx + int(post_act)
        return self.network[:network_idx+1](x)
    
    def forward(self, x):
        if self.classification:
            y=F.softmax(self.network(x), dim=-1)
        else:
            y= self.network(x)
        return y
    
    def predict_with_logits(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        # Currently used by SHAP
        input = x if torch.is_tensor(x) else torch.from_numpy(np.array(x))
        return self.forward(input.float()).detach().numpy()
    
    def predict(self, x, argmax=False):
        # Currently used by LIME
        input = torch.squeeze(x) if torch.is_tensor(x) else torch.from_numpy(np.array(x))
        output = self.forward(input.float()).detach().numpy()
        return output.argmax(axis=-1) if argmax else output

class ANNLightning(pl.LightningModule):
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
        else:
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            self.test_metrics = {
                "rmse": rmse
            }

        print("Test Metrics:", self.test_metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def checkpoint(self):
        return ModelCheckpoint(save_top_k=1,mode='min',save_last=False,monitor=self.monitor)


if __name__ == "__main__":
    from utils import CONFIG
    config = CONFIG['syn']
    model = ANN(config['num_features'],config['hidden_features_ann'],classification=config['classification'])
    print(model(torch.rand((10,config['num_features']))))