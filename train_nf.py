from flow import flow,CausalNF
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
    flow_ = flow(config['num_features'],adj,config['hidden_features_flow'])
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
    logger.experiment.add_text('hidden_featues_flow',str(config['hidden_features_flow']),global_step=0)
    logger.experiment.add_text('Architecture',str(scm),global_step=0)
    early_stopping_callback = EarlyStopping(
                        monitor='validation_loss',  
                        patience=50,          
                        mode='min',           
                        verbose=True          
                        )

    use_gpu = torch.cuda.is_available()

    trainer = pl.Trainer(
                default_root_dir='lightning_logs',
                accelerator='gpu' if use_gpu else 'cpu',
                devices=1,  # <- always 1, even for CPU
                callbacks=[scm.checkpoint(), early_stopping_callback, vis_callback],
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





if __name__ == "__main__":
    name = 'german'
    config = CONFIG[name]
    train_NF(config)
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
        
        
