import os
import pickle
import numpy as np
import pandas as pd
from time import time

start_time = time()

os.chdir(os.path.dirname(__file__))

df_varlist = pd.read_csv('Neonatal Mortality Predictor List.csv')

df_2016 = pd.read_csv('http://www.nber.org/natality/2016/natl2016.csv.zip', 
                      usecols=df_varlist['feature'],
                      dtype=pd.Series(df_varlist['dtype'].values,
                                      index=df_varlist['feature']).to_dict())
df_2016 = df_2016[df_2016['ilive'] != 'U']
df_2016['ilive'] = df_2016['ilive'] == 'N'
    
for var_name in df_2016.select_dtypes(include=[np.number]):
    var_max = df_2016[var_name].max()
    if var_max in [99, 999, 9999]:
        df_2016[var_name].replace(var_max, np.nan, inplace=True)
    var_max = df_2016[var_name].max()
    if var_max == 888:
        df_2016[var_name].replace(var_max, np.nan, inplace=True)
            
cat_ranks = {}
for var_name in df_2016.select_dtypes(include=['category']):
    cat_ranks[var_name] = df_2016.groupby([var_name])['ilive'].mean().to_frame()
    cat_ranks[var_name]['rank_'+var_name] = cat_ranks[var_name]['ilive'].rank(method='dense')
    cat_ranks[var_name] = pd.to_numeric(cat_ranks[var_name]['rank_'+var_name], downcast='unsigned')
    cat_ranks[var_name] = cat_ranks[var_name].to_dict()
    df_2016[var_name] = df_2016[var_name].map(cat_ranks[var_name])
        
with open('cat_ranks.p', 'wb') as file:
    pickle.dump(cat_ranks, file, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('df_2016.p', 'wb') as file:
    pickle.dump(df_2016, file, protocol=pickle.HIGHEST_PROTOCOL)
    
del df_2016

df_2015 = pd.read_csv('http://www.nber.org/natality/2015/natl2015.csv.zip',
                      usecols=df_varlist['feature'],
                      dtype=pd.Series(df_varlist['dtype'].values,
                                      index=df_varlist['feature']).to_dict())
df_2015 = df_2015[df_2015['ilive'] != 'U']
df_2015['ilive'] = df_2015['ilive'] == 'N'

for var_name in df_2015.select_dtypes(include=[np.number]):
    var_max = df_2015[var_name].max()
    if var_max in [99, 999, 9999]:
        df_2015[var_name].replace(var_max, np.nan, inplace=True)
    var_max = df_2015[var_name].max()
    if var_max == 888:
        df_2015[var_name].replace(var_max, np.nan, inplace=True)

for var_name in df_2015.select_dtypes(include=['category']):
    df_2015[var_name] = df_2015[var_name].map(cat_ranks[var_name])
    
with open('df_2015.p', 'wb') as file:
    pickle.dump(df_2015, file, protocol=pickle.HIGHEST_PROTOCOL)
    
print('Run time: ' + str(time() - start_time) + ' seconds')