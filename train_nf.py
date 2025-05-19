from flow import flow,CausalNF,num_features,adjacency
import pytorch_lightning as pl
from pytorch_lightning.callbacks import  EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from DataLoaders import ReturnDataloaders

seeds = [10,20,30]

for model_name in num_features.keys():
    for seed in seeds:   
        torch.manual_seed(seed)
        pl.seed_everything(seed)

        trainloader, testloader = ReturnDataloaders(dataset=model_name,batch_size=32,seed=seed)

        flow_ = flow(num_features[model_name],adjacency[model_name])
        scm = CausalNF(flow=flow_, lr = 3e-4)

        run_name = f"{model_name}_nf"
        version_name = f"{model_name}_nf_seed_{seed}"

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

        trainer.fit(model=scm, train_dataloaders= trainloader, val_dataloaders=testloader)
        
