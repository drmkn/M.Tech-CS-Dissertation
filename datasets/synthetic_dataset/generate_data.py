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
    features[:, 0] = np.random.rand(num_samples)  # P
    features[:, 1] = 2*features[:, 0] + np.sqrt(0.1)*np.random.randn(num_samples)  # Q
    features[:, 2] = 2*features[:, 0] - features[:, 1] + np.sqrt(0.1)*np.random.randn(num_samples)  # R
    return features

def generate_output(features):
    return (features[:, 2])**2 + np.sqrt(np.exp(features[:, 1])) + np.sqrt(0.1)*np.random.randn(features.shape[0]) #S

N = train_samples+test_samples
features = generate_features(train_samples+test_samples)
outputs = generate_output(features)
df = pd.DataFrame(data = {'P':features[:,0], 'Q':features[:,1], 'R':features[:,2], 'S' : outputs})
df = convert_to_cont(df,config['discrete_cols'],config['seed'])
# df_x = df[config['var_names']]
# df_y = df[config['target']]
train_df, test_df = train_test_split(df, train_size=config['train_samples']/(config['train_samples']+config['test_samples']), random_state=config['seed'])
scaler = MinMaxScaler()
train_df = pd.DataFrame(scaler.fit_transform(train_df),columns=config['var_names'] + ['S'])
# train_df['S'] = train_y.values
test_df = pd.DataFrame(scaler.transform(test_df),columns=config['var_names'] + ['S'])
# test_df['S'] = test_y.values
train_path = os.path.join(script_dir, 'syn-train.csv')
test_path = os.path.join(script_dir, 'syn-test.csv')

train_df.to_csv(train_path,index = False)
test_df.to_csv(test_path,index = False)

