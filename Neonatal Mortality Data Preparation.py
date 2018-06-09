import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from time import time

start_time = time()

os.chdir(os.path.dirname(__file__))

def load_and_clean(year, df_varlist):
    url = 'http://www.nber.org/natality/' + year + '/natl' + year + '.csv.zip'
    df = pd.read_csv(url,
                     usecols=df_varlist['feature'],
                     dtype=pd.Series(df_varlist['dtype'].values,
                                     index=df_varlist['feature']).to_dict())
    print(year + ' Counts of Live Births by Vital Status:')
    print(df['ilive'].value_counts())
    print(year + ' Counts of NICU-admitted Live Births by Vital Status:')
    print(df[df['ab_nicu'] == 'Y']['ilive'].value_counts())
    df = df[df['ilive'] != 'U']
    df['ilive'] = df['ilive'] == 'N'
    df = missing_to_nan(df)
    print(year + ' Correlation Between Gestation Duration Measures:')
    print(df['oegest_comb'].corr(df['combgest']))
    return df

def missing_to_nan(df):
    for var_name in df.select_dtypes(include=[np.number]):
        var_max = df[var_name].max()
        if var_max in [99, 999, 9999]:
            df[var_name].replace(var_max, np.nan, inplace=True)
        var_max = df[var_name].max()
        if var_max == 888:
            df[var_name].replace(var_max, np.nan, inplace=True)
    return df

def target_encode(df, file_name):
    Xy_train, Xy_valid = train_test_split(df, test_size=0.25, random_state=9999)
    Xy_valid, Xy_hvalid = train_test_split(Xy_valid, test_size=0.5, random_state=9999)
    train_mean = Xy_train['ilive'].mean()
    cat_ranks = {}
    for var_name in Xy_train.select_dtypes(include=['category']):
        cat_ranks[var_name] = Xy_train.groupby([var_name])['ilive'].agg(['mean', 'count'])
        cat_ranks[var_name]['weight'] = (train_mean*(1-train_mean)*cat_ranks[var_name]['count']
                                            / (train_mean*(1-train_mean)*cat_ranks[var_name]['count']
                                               + cat_ranks[var_name]['mean']*(1-cat_ranks[var_name]['mean'])))
        cat_ranks[var_name]['shrunk_mean'] = (cat_ranks[var_name]['mean']*cat_ranks[var_name]['weight']
                                              + train_mean*(1 - cat_ranks[var_name]['weight']))
        cat_ranks[var_name]['rank_'+var_name] = cat_ranks[var_name]['shrunk_mean'].rank(method='dense')
        cat_ranks[var_name] = pd.to_numeric(cat_ranks[var_name]['rank_'+var_name], downcast='unsigned')
        cat_ranks[var_name] = cat_ranks[var_name].to_dict()
        for df in [Xy_train, Xy_valid, Xy_hvalid]:
            df[var_name] = df[var_name].map(cat_ranks[var_name])
    with open(file_name + ' Target Mean Encodings.p', 'wb') as file:
        pickle.dump(cat_ranks, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_name + ' Training Set.p', 'wb') as file:
        pickle.dump(Xy_train, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_name + ' Validation Set.p', 'wb') as file:
        pickle.dump(Xy_valid, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_name + ' Hypervalidation Set.p', 'wb') as file:
        pickle.dump(Xy_hvalid, file, protocol=pickle.HIGHEST_PROTOCOL)
    return cat_ranks

pd.set_option('precision', 5)

df_varlist = pd.read_csv('Neonatal Mortality Predictor List.csv')

df_2016 = load_and_clean('2016', df_varlist)

df_2016_NICU = df_2016[df_2016['ab_nicu'] == 'Y'].drop(['ab_nicu'], axis=1)
cat_ranks_NICU = target_encode(df_2016_NICU, 'NICU 2016')
del df_2016_NICU

df_2016 = df_2016.drop(['ab_nicu'], axis=1)
cat_ranks = target_encode(df_2016, 'Full 2016')
del df_2016

df_2015 = load_and_clean('2015', df_varlist)
        
df_2015_NICU = df_2015[df_2015['ab_nicu'] == 'Y'].drop(['ab_nicu'], axis=1)
for var_name in df_2015_NICU.select_dtypes(include=['category']):
    df_2015_NICU[var_name] = df_2015_NICU[var_name].map(cat_ranks_NICU[var_name])
with open('NICU 2015.p', 'wb') as file:
    pickle.dump(df_2015_NICU, file, protocol=pickle.HIGHEST_PROTOCOL)
del df_2015_NICU

df_2015 = df_2015.drop(['ab_nicu'], axis=1)
for var_name in df_2015.select_dtypes(include=['category']):
    df_2015[var_name] = df_2015[var_name].map(cat_ranks[var_name])
with open('Full 2015.p', 'wb') as file:
    pickle.dump(df_2015, file, protocol=pickle.HIGHEST_PROTOCOL)
    
print('Run time: ' + str(time() - start_time) + ' seconds')