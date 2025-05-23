import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from utils import MMD
import networkx as nx
from utils import PREFIX

num_features = {'syn':3, 'auto_mpg':5, 'compas':7}

meta_datas = {'syn':['c']*3,
              'auto_mpg': ['c']*5,
              'compas': ['c']*7 }

adjacency = {'syn':torch.tensor([[1,1,1],
                                 [0,1,1],
                                 [0,0,1]
                                 ]),
             'auto_mpg': torch.tensor([[1,1,0,1,0],
                                       [0,1,1,0,1],
                                       [0,0,1,0,1],
                                       [0,0,0,1,1],
                                       [0,0,0,0,1],
                                        ]),
             'compas': torch.tensor([[1,1,1,1,1,0,0],
                                     [0,1,0,0,0,0,0],
                                     [0,1,1,1,1,0,0],
                                     [0,1,0,1,0,0,0],
                                     [0,0,0,1,1,0,0],
                                     [0,1,1,1,1,1,0],
                                     [0,1,1,1,1,0,1]
                                        ])                              
            }

causal_graphs = {'syn' : nx.DiGraph([ (0,1),(0,2),
                                     (1,2)
                                    ]),
                 'compas' : nx.DiGraph([ (0,1),(0,2),(0,3),(0,4),
                            (2,1), (2,3), (2,4),
                            (3,1),
                            (4,3),
                            (5,1),(5,2),(5,3),(5,4),
                            (6,1),(6,2),(6,3),(6,4)
                           ]),
                  'auto_mpg' : nx.DiGraph([ (0,1),(0,3),
                                           (1,2),(1,4),
                                           (2,4),
                                           (3,4),
                            ])                                     

                }

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

#----------------------------------------------

import sys
# sys.path.append(PREFIX+'Dissertation/zuko')
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.flows import Flow, UnconditionalDistribution, NSF, UnconditionalTransform
from zuko.distributions import BoxUniform, DiagNormal
from zuko.transforms import SigmoidTransform

def flow(num_features, adjacency,hidden_features=[64,64]):

    
    flow_=  Flow(
        transform=[MaskedAutoregressiveTransform(features=num_features, 
                                        hidden_features=hidden_features,adjacency = adjacency,
                                        activation =torch.nn.LeakyReLU
                                    ),
                   UnconditionalTransform(SigmoidTransform),
        ], 
        base = UnconditionalDistribution(
        #            BoxUniform,
        #            torch.zeros(num_features),
        #            torch.ones(num_features)
                #    buffer=True,) if base_ == 'uniform' else 
                    DiagNormal, ##normal base
                    torch.zeros(num_features),
                    torch.ones(num_features),
                    buffer=True
                ) 
    
    )
    return flow_


if __name__ == "__main__":
    
    pass