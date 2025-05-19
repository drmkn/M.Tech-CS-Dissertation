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
    def __init__(self, train_path: str , test_path :str, batch_size: int = 32):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load and split data
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        dataset = CustomDataset(df)

        train_len = int(len(dataset) * self.split_ratio)
        val_len = len(dataset) - train_len
        self.train_set, self.val_set = random_split(dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

