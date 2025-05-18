'''Estimate DAG of a given dataset using notears algorithm'''
import torch
import sys
import os 
sys.path.append('/home/saptarshi/Dhruv/Dissertation/notears')
from notears.nonlinear import notears_nonlinear,NotearsMLP
import pandas as pd
import numpy as np
from utils import CONFIG

config = CONFIG['syn']
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.set_default_dtype(torch.double)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv(filepath_or_buffer='./datasets/synthetic_dataset/syn-train.csv')
X = torch.tensor(data.to_numpy(), dtype=torch.double).to(device)

model = NotearsMLP(dims=[data.shape[1],5, 1]).to(device)

dag = notears_nonlinear(model=model, X=X.cpu().numpy(),lambda1=0.01)  # notears_nonlinear still expects numpy
print(dag)

