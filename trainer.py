from architectures import MLP,flow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import  EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
from dataloader import CustomDataModule
from utils import CONFIG,get_adjacency
import os
from kde_visualisation_callback import SampleVisualizationCallback
from utils import log_adjacency_as_text
from pl_modules import MLPLightning,CausalNF


#------------------------------MLPtrainer--------------------------------------------#
def train_MLP(config):
    torch.set_float32_matmul_precision('high')
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

    dm = CustomDataModule(config)
    dm.setup() ##setup train test data
    
    model = MLP(input_dim=config['num_features'],hidden_layers=config['hidden_layers_mlp'],
                classification=config['classification'])
    model_pl = MLPLightning(model=model,lr = 3e-4)

    run_name = f"{config['name']}_mlp"
    version_name = f"{config['name']}_mlp_seed_{config['seed']}"
    base_dir = os.path.join("models",run_name,version_name)
    os.makedirs(base_dir, exist_ok=True)
    logger = TensorBoardLogger(
        save_dir=base_dir,
        name="logs"
    )
    logger.experiment.add_text('batch_size',str(config['batch_size']),global_step=0)
    logger.experiment.add_text('hidden_layers_mlp',str(config['hidden_layers_mlp']),global_step=0)
    logger.experiment.add_text('Architecture',str(model_pl),global_step=0)
    early_stopping_callback = EarlyStopping(
                        monitor='validation_loss',  
                        patience=20,          
                        mode='min',           
                        verbose=True          
                        )

    use_gpu = torch.cuda.is_available()

    trainer = pl.Trainer(
                default_root_dir='lightning_logs',
                accelerator='gpu' if use_gpu else 'cpu',
                devices=1,  # <- always 1, even for CPU
                callbacks=[model_pl.checkpoint(),early_stopping_callback],
                max_epochs=500,
                fast_dev_run=False,
                precision="16-mixed" if use_gpu else 32,
                reload_dataloaders_every_n_epochs=10,
                logger=logger
            )

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    trainer.fit(model=model_pl, datamodule=dm) 
    trainer.test(datamodule=dm, ckpt_path='best' )

#------------------------------MLPtrainer--------------------------------------------#


#------------------------------CNFtrainer--------------------------------------------#

def train_NF(config,ground_truth_dag = True):
    torch.set_float32_matmul_precision('high')
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

    dm = CustomDataModule(config)
    dm.setup() ##setup train test data
    if ground_truth_dag:
        adj = config['gd_adjacency']
        run_name = f"{config['name']}_nf"
    else:
        adj = get_adjacency(config)
        run_name = f"{config['name']}_nf_notears_dag"
    flow_ = flow(config['num_features'],adj,config['hidden_layers_flow'])
    scm = CausalNF(flow=flow_, lr = 3e-4)

    vis_callback = SampleVisualizationCallback(config=config)
    version_name = f"{config['name']}_nf_seed_{config['seed']}"
    base_dir = os.path.join("models",run_name,version_name)
    os.makedirs(base_dir, exist_ok=True)
    logger = TensorBoardLogger(
        save_dir=base_dir,
        name="logs"
    )
    log_adjacency_as_text(logger,adj,config['var_names'])
    logger.experiment.add_text('batch_size',str(config['batch_size']),global_step=0)
    logger.experiment.add_text('hidden_layers_flow',str(config['hidden_layers_flow']),global_step=0)
    logger.experiment.add_text('Architecture',str(scm),global_step=0)
    early_stopping_callback = EarlyStopping(
                        monitor='validation_loss',  
                        patience=20,          
                        mode='min',           
                        verbose=True          
                        )

    use_gpu = torch.cuda.is_available()

    trainer = pl.Trainer(
                default_root_dir='lightning_logs',
                accelerator='gpu' if use_gpu else 'cpu',
                devices=1,  # <- always 1, even for CPU
                callbacks=[scm.checkpoint(),early_stopping_callback, vis_callback],
                max_epochs=500,
                fast_dev_run=False,
                precision="16-mixed" if use_gpu else 32,
                reload_dataloaders_every_n_epochs=10,
                logger=logger
            )

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    trainer.fit(model=scm, datamodule=dm) 
    trainer.test(datamodule=dm, ckpt_path='best' )

#------------------------------CNFtrainer--------------------------------------------#


if __name__ == "__main__":
    for name in ['syn']:
        for seed in [1,2,3]:
            CONFIG[name]['seed'] = seed
            config = CONFIG[name]
            train_MLP(config=config)
            # train_NF(config=config,ground_truth_dag=True)