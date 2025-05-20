import pandas as pd
from sklearn.model_selection import train_test_split
from utils import CONFIG
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
name = 'german'
config = CONFIG[name]
df = pd.read_csv("/home/saptarshi/Dhruv/Dissertation/datasets/german_credit/german_credit_data.csv")
df = df.drop(columns=['Index'])
df.rename(columns={'Sex' : 'G', 'Age' : 'A', 'Credit amount':'C',
                    'Duration' : 'D', 'Default' : 'R'},inplace=True)
df['G'] = df['G'].map({'male': 0, 'female': 1})
df['R'] = df['R'].map({'good': 0, 'bad': 1})
df = df.astype(float)
df = (df - df.min())/(df.max()-df.min())
train_df, test_df = train_test_split(df, train_size=config['train_samples']/(config['train_samples']+config['test_samples']), random_state=config['seed'])
train_path = os.path.join(script_dir, 'german-train.csv')
test_path = os.path.join(script_dir, 'german-test.csv')

train_df.to_csv(train_path,index = False)
test_df.to_csv(test_path,index = False)

