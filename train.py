
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
pd.set_option('display.max_columns', None)


train_df = pd.read_csv('data/dataTrain.csv')
no_label_df = pd.read_csv('data/dataNoLabel.csv')
test_df = pd.read_csv('data/dataA.csv')
train_df['f3'] = train_df['f3'].map({'low':1,'mid':2,'high':3})


def col_zero_ratio(df):
    df_len = len(df)
    zero_stat = pd.DataFrame()
    stat_cols = [col for col in df.columns if col not in ['id','label']]
    zero_stat['stat_col'] = stat_cols
    zero_stat['zero_ratio'] = ((df[stat_cols]==0).sum(axis=0)/df_len).values
    return zero_stat


train_df_zero = col_zero_ratio(train_df)
no_label_df_zero = col_zero_ratio(no_label_df)
test_df_zero = col_zero_ratio(test_df)


data_zero_stat = pd.merge(train_df_zero, no_label_df_zero,how='left',on=['stat_col'],suffixes=('','_no_label'))\
      .merge(test_df_zero,how='left',on=['stat_col'],suffixes=('','_test'))
data_zero_stat['train-train_nolabel'] = data_zero_stat['zero_ratio'] - data_zero_stat['zero_ratio_no_label']
data_zero_stat['train-test'] = data_zero_stat['zero_ratio'] - data_zero_stat['zero_ratio_test']

rm_cols = data_zero_stat[data_zero_stat['zero_ratio'] >= 0.99]['stat_col'].values

fea_cols = [col for col in train_df.columns if col not in rm_cols]
print(train_df[fea_cols])











