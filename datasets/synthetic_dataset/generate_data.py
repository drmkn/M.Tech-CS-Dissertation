import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from utils import CONFIG

script_dir = os.path.dirname(os.path.abspath(__file__))
config = CONFIG['syn']

seed = config['seed']
np.random.seed(seed)
train_samples = config['train_samples']
test_samples = config['test_samples']

def generate_features(num_samples):
    features = np.zeros((num_samples, 3))
    features[:, 0] = np.random.rand(num_samples)  # W
    features[:, 1] = features[:, 0] / 2 + 0.1*np.random.randn(num_samples)  # Z
    features[:, 2] = -features[:, 0] - features[:, 1] + 0.1*np.random.randn(num_samples)  # X
    return features

def generate_output(features):
    return features[:, 2]**3 + np.log(features[:, 1]**2) + 0.1*np.random.randn(features.shape[0])

N = train_samples+test_samples
features = generate_features(train_samples+test_samples)
outputs = generate_output(features)
df = pd.DataFrame(data = {'W':features[:,0], 'Z':features[:,1], 'X':features[:,2], 'Y' : outputs})

train_df, test_df = train_test_split(df, train_size=train_samples/N, random_state=seed)

train_path = os.path.join(script_dir, 'syn-train.csv')
test_path = os.path.join(script_dir, 'syn-test.csv')

train_df.to_csv(train_path,index = False)
test_df.to_csv(test_path,index = False)

