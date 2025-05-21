import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

class CustomDataset(Dataset):
    def __init__(self, dataframe,config):
        self.X = torch.tensor(dataframe.drop(columns=config['target']).values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[config['target']].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config :dict, batch_size: int = 64):
        super().__init__()
        self.config = config
        self.train_path = config['train_data']
        self.test_path = config['test_data']
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load data
        self.train_dataset = CustomDataset(pd.read_csv(self.train_path),self.config)
        self.test_dataset = CustomDataset(pd.read_csv(self.test_path),self.config)
        self.val_dataset = CustomDataset(pd.read_csv(self.test_path),self.config)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size//2, shuffle=False,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config['test_samples'], shuffle= False,num_workers=4)

