from flow import flow,CausalNF
import pytorch_lightning as pl
from pytorch_lightning.callbacks import  EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
from dataloader import CustomDataModule
from utils import CONFIG,get_adjacency
import os
name = 'syn'
config = CONFIG[name]

def train_NF(config):
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

    dm = CustomDataModule(config)
    dm.setup() ##setup train test data
    adjacency = get_adjacency(config)
    flow_ = flow(config['num_features'],adjacency)
    scm = CausalNF(flow=flow_, lr = 3e-4)

    run_name = f"{config['name']}_nf"
    version_name = f"{config['name']}_nf_seed_{seed}"

    os.makedirs('models',exist_ok=True)
    logger = TensorBoardLogger('models', name=run_name,version=version_name)

    early_stopping_callback = EarlyStopping(
                        monitor='validation_loss',  
                        patience=10,          
                        mode='min',           
                        verbose=True          
                        )

    trainer = pl.Trainer(default_root_dir = 'lightning_logs',
                        devices=torch.cuda.device_count(),
                        callbacks= [scm.checkpoint(), early_stopping_callback],
                        max_epochs=1000,
                        fast_dev_run=False,
                        precision="16-mixed",
                        reload_dataloaders_every_n_epochs=10,
                        logger = logger
                        )

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    trainer.fit(model=scm, datamodule=dm) 



if __name__ == "__main__":
    name = 'syn'
    config = CONFIG[name]
    train_NF(config)
        
        
