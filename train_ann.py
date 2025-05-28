import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
from dataloader import CustomDataModule
from utils import CONFIG
import os
from ann import ANN,ANNLightning


def train_ANN(config):
    torch.set_float32_matmul_precision('high')
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

    dm = CustomDataModule(config)
    dm.setup() ##setup train test data
    
    model = ANN(input_dim=config['num_features'],hidden_layers=config['hidden_features_ann'],
                classification=config['classification'])
    model_pl = ANNLightning(model = model, lr = 1e-3)

    run_name = f"{config['name']}_ann"
    version_name = f"{config['name']}_ann_seed_{config['seed']}"
    base_dir = os.path.join("models",run_name,version_name)
    os.makedirs(base_dir, exist_ok=True)
    logger = TensorBoardLogger(
        save_dir=base_dir,
        name="logs"
    )
    logger.experiment.add_text('batch_size',str(config['batch_size']),global_step=0)
    logger.experiment.add_text('hidden_featues_ann',str(config['hidden_features_ann']),global_step=0)
    logger.experiment.add_text('Architecture',str(model_pl),global_step=0)
    early_stopping_callback = EarlyStopping(
                        monitor='validation_loss',  
                        patience=50,          
                        mode='min',           
                        verbose=True          
                        )

    trainer = pl.Trainer(default_root_dir = 'lightning_logs',
                        devices=torch.cuda.device_count(),
                        callbacks= [model_pl.checkpoint(),early_stopping_callback],
                        max_epochs=1000,
                        fast_dev_run=False,
                        precision="16-mixed",
                        reload_dataloaders_every_n_epochs=10,
                        logger = logger
                        )

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    trainer.fit(model=model_pl, datamodule=dm) 
    trainer.test(datamodule=dm, ckpt_path='best' )





if __name__ == "__main__":
    name = 'syn'
    config = CONFIG[name]
    train_ANN(config)
    # ckt = '/home/saptarshi/Dhruv/Dissertation/models/syn_nf/syn_nf_seed_10/checkpoints/epoch=250-step=15813.ckpt'
    # flow_ = flow(3,get_adjacency(config))
    # scm_fit = CausalNF.load_from_checkpoint(checkpoint_path = ckt, flow=flow_, lr=3e-4)
    # scm_fit.eval()
    # flow_ = scm_fit.flow()
    # # print(scm_fit)
    # # print(flow_)
    # base_sample = flow_.base.sample((config['test_samples'],))
    # print(base_sample)
    # x = flow_.transform(base_sample)
    # print(x)
        
        
