import pandas as pd
from sklearn.model_selection import train_test_split
from utils import CONFIG,convert_to_cont,PREFIX
import os
from sklearn.preprocessing import MinMaxScaler
script_dir = os.path.dirname(os.path.abspath(__file__))
config = CONFIG['cancer']
df = pd.read_csv(PREFIX + "datasets/lung_cancer/data.csv")
df.rename(columns={'asia' : 'A','tub' : 'T','smoke' : 'S', 'lung' : 'L',
                   'bronc' : 'B', 'either' : 'E', 'xray' : 'X', 'dysp' : 'D'},inplace=True)
df = convert_to_cont(df,config['discrete_cols'],config['seed'])
train_df, test_df = train_test_split(df, train_size=config['train_samples']/(config['train_samples']+config['test_samples']), random_state=config['seed'])
scaler = MinMaxScaler()
train_df = pd.DataFrame(scaler.fit_transform(train_df),columns=config['var_names'] + ['D'])
test_df = pd.DataFrame(scaler.transform(test_df),columns=config['var_names']+['D'])
train_path = os.path.join(script_dir, 'cancer-train.csv')
test_path = os.path.join(script_dir, 'cancer-test.csv')

train_df.to_csv(train_path,index = False)
test_df.to_csv(test_path,index = False)

