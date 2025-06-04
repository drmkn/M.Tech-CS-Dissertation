import pandas as pd
from sklearn.model_selection import train_test_split
from utils import CONFIG,convert_to_cont,PREFIX
import os
from sklearn.preprocessing import MinMaxScaler
script_dir = os.path.dirname(os.path.abspath(__file__))
name = 'german'
config = CONFIG[name]
df = pd.read_csv(PREFIX +"datasets/german_credit/german_credit_data.csv")
df = df.drop(columns=['Index'])
df.rename(columns={'Sex' : 'G', 'Age' : 'A', 'Credit amount':'C',
                    'Duration' : 'D', 'Default' : 'R'},inplace=True)
df['G'] = df['G'].map({'male': 0, 'female': 1})
df['R'] = df['R'].map({'good': 0, 'bad': 1})
df = convert_to_cont(df,config['discrete_cols'],config['seed'])
train_df, temp_df = train_test_split(df, train_size=config['train_samples']/(config['train_samples']+config['test_samples']+config['val_samples']), random_state=config['seed'])
test_df, val_df = train_test_split(temp_df, train_size=config['test_samples']/(config['test_samples']+config['val_samples']), random_state=config['seed'])
scaler = MinMaxScaler()
train_df = pd.DataFrame(scaler.fit_transform(train_df),columns=config['var_names'] + ['R'])
test_df = pd.DataFrame(scaler.transform(test_df),columns=config['var_names']+['R'])
val_df = pd.DataFrame(scaler.transform(val_df),columns=config['var_names']+['R'])
train_path = os.path.join(script_dir, 'german-train.csv')
test_path = os.path.join(script_dir, 'german-test.csv')
val_path = os.path.join(script_dir, 'german-val.csv')

train_df.to_csv(train_path,index = False)
test_df.to_csv(test_path,index = False)
val_df.to_csv(val_path,index = False)

