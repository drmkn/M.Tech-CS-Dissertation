import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

class CustomDataset(Dataset):
    def __init__(self, dataframe,config):
        self.X = torch.tensor(dataframe.drop(columns=config['target']).values, dtype=torch.float32)
        if config['classification']:
            self.y = torch.tensor(dataframe[config['target']].values, dtype=torch.long).squeeze()
        else:
            self.y = torch.tensor(dataframe[config['target']].values, dtype=torch.float32).squeeze()    
        # print(self.X)
        # print(self.y)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config :dict):
        super().__init__()
        self.config = config
        self.train_path = config['train_data']
        self.test_path = config['test_data']
        self.batch_size = config['batch_size']

    def setup(self, stage=None):
        # Load data
        self.train_dataset = CustomDataset(pd.read_csv(self.train_path),self.config)
        self.test_dataset = CustomDataset(pd.read_csv(self.test_path),self.config)
        self.val_dataset = CustomDataset(pd.read_csv(self.test_path),self.config)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['test_samples'], shuffle=False,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config['test_samples'], shuffle= False,num_workers=4)

if __name__ == "__main__":
    from utils import CONFIG
    config = CONFIG['syn']
    d = CustomDataset(pd.read_csv(config['train_data']),config)
