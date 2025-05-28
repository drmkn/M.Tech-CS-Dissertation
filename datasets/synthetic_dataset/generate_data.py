import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from utils import CONFIG,convert_to_cont

script_dir = os.path.dirname(os.path.abspath(__file__))
config = CONFIG['syn']

seed = config['seed']
np.random.seed(seed)
train_samples = config['train_samples']
test_samples = config['test_samples']

def generate_features(num_samples):
    features = np.zeros((num_samples, 3))
    features[:, 0] = np.random.rand(num_samples)  # W
    features[:, 1] = 3*features[:, 0] + np.sqrt(0.1)*np.random.randn(num_samples)  # Z
    features[:, 2] = 3*features[:, 0] - features[:, 1] + np.sqrt(0.1)*np.random.randn(num_samples)  # X
    return features

def generate_output(features):
    return 4*features[:, 2] + np.exp(4*features[:, 1]) + np.sqrt(0.1)*np.random.randn(features.shape[0])

N = train_samples+test_samples
features = generate_features(train_samples+test_samples)
outputs = generate_output(features)
df = pd.DataFrame(data = {'W':features[:,0], 'Z':features[:,1], 'X':features[:,2], 'Y' : outputs})
df = convert_to_cont(df,config['discrete_cols'],config['seed'])
train_df, test_df = train_test_split(df, train_size=config['train_samples']/(config['train_samples']+config['test_samples']), random_state=config['seed'])
scaler = MinMaxScaler()
train_df = pd.DataFrame(scaler.fit_transform(train_df),columns=config['var_names'] + ['Y'])
test_df = pd.DataFrame(scaler.transform(test_df),columns=config['var_names']+['Y'])
train_path = os.path.join(script_dir, 'syn-train.csv')
test_path = os.path.join(script_dir, 'syn-test.csv')

train_df.to_csv(train_path,index = False)
test_df.to_csv(test_path,index = False)

