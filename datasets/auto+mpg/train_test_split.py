import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from utils import CONFIG
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
config = CONFIG['mpg']
data = pd.read_csv('/home/saptarshi/Dhruv/Dissertation/datasets/auto+mpg/auto-mpg.csv')
data.drop(columns = ['car name','model year','origin'],inplace=True)
data = data[~data.isin(['?']).any(axis=1)]
data.rename(columns={'cylinders':'C','mpg':'M','displacement':'D',
                     'horsepower' : 'H','weight':'W','acceleration':'A'},inplace=True)
# print(data.dtypes)
data['H'] = data['H'].astype('float64')
print(data.dtypes)
# data = (data - data.min()) / (data.max() - data.min()) #normalisation
print(data.shape)
seed = config['seed']
np.random.seed(seed)
train_samples = config['train_samples']
test_samples = config['test_samples']
N = train_samples+test_samples

train_df, test_df = train_test_split(data, train_size=train_samples/N, random_state=seed)

train_path = os.path.join(script_dir, 'mpg-train.csv')
test_path = os.path.join(script_dir, 'mpg-test.csv')

train_df.to_csv(train_path,index = False)
test_df.to_csv(test_path,index = False)

print(train_df.shape,test_df.shape)

