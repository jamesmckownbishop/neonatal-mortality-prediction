import os
from functools import partial
from math import log
import pickle
import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
import sklearn.metrics
import xgboost
from hyperopt import fmin, tpe, hp

start_time = time()

os.chdir(os.path.dirname(__file__))

def train_model(params, X, y):
    params.update({
                   'objective':'binary:logistic',
                   'eval_metric':'auc',
                   'seed':9999,
                   'tree_method':'hist',
                   'grow_policy':'lossguide'
                  })
    params['max_leaves'] = int(params['max_leaves'])
    params['max_bin'] = int(params['max_bin'])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=9999)
    X_test, X_valid, y_test, y_valid = train_test_split(X_valid, y_valid, test_size=0.5, random_state=9999)
    dtrain = xgboost.DMatrix(X_train, y_train)
    dvalid = xgboost.DMatrix(X_valid, y_valid)
    del X_train, X_valid, y_train
    model = xgboost.train(params, dtrain, 1024, evals=[(dvalid, 'eval')], early_stopping_rounds=16)
    dtest = xgboost.DMatrix(X_test, y_test)
    del X_test
    y_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    auroc = sklearn.metrics.roc_auc_score(y_test, y_pred)
    return 1-auroc

def choose_model(X, y, spec_name):
    best = fmin(fn=partial(train_model,
                           X=X,
                           y=y),
                space=space,
                algo=tpe.suggest,
                max_evals=256,
                rstate=np.random.RandomState(9999))
    best['max_leaves'] = 128 - best['max_leaves']
    best['max_bin'] = 2 + best['max_bin']
    best['subsample'] = 1.5 - best['subsample']
    best['colsample_bytree'] = 1.5 - best['colsample_bytree']
    best['colsample_bylevel'] = 1.5 - best['colsample_bylevel']
    with open(spec_name + '_best_params.p', 'wb') as file:
        pickle.dump(best, file, protocol=pickle.HIGHEST_PROTOCOL)
    return best

df_2016 = pickle.load(open("df_2016.p", "rb"))
df_varlist = pd.read_csv('Neonatal Mortality Predictor List.csv')
pre_birth_vars = df_varlist[df_varlist['pre_birth']==1]['feature'].tolist()
del df_varlist

space = {
         'objective':'binary:logistic',
         'eval_metric':'auc',
         'seed':9999,
         'tree_method':'hist',
         'grow_policy':'lossguide',
         'max_delta_step':hp.lognormal('max_delta_step', 0, 1),
         'min_child_weight':hp.lognormal('min_child_weight', 0, 1),
         'gamma':hp.lognormal('gamma', 0, 1),
         'lambda':hp.lognormal('lambda', 0, 1),
         'alpha':hp.lognormal('alpha', 0, 1),
         'eta':hp.loguniform('eta', log(2**-7), 0),
         'max_leaves':128 - hp.qloguniform('max_leaves', log(4), log(128), 1),
         'max_bin':2 + hp.qlognormal('max_bin', log(256 - 2), 1, 1),
         'subsample':1.5 - hp.loguniform('subsample', log(0.5), 0),
         'colsample_bytree':1.5 - hp.loguniform('colsample_bytree', log(0.5), 0),
         'colsample_bylevel':1.5 - hp.loguniform('colsample_bylevel', log(0.5), 0)
        }

best_pre_birth = choose_model(df_2016[pre_birth_vars],
                              df_2016['ilive'],
                              'pre_birth')

best_NICU = choose_model(df_2016[df_2016['ab_nicu']==2].drop(['ilive', 'ab_nicu'], axis=1),
                         df_2016[df_2016['ab_nicu']==2]['ilive'],
                         'NICU')

best_post_birth = choose_model(df_2016.drop(['ilive', 'ab_nicu'], axis=1),
                                            df_2016['ilive'],
                                            'post_birth')
    
print('Run time: ' + str(time() - start_time) + ' seconds')