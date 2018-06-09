import os
from functools import partial
from math import log
import pickle
import numpy as np
import pandas as pd
from time import time
import sklearn.metrics

import xgboost
from hyperopt import fmin, tpe, hp

# Run time: 32740.427926301956 seconds

start_time = time()

os.chdir(os.path.dirname(__file__))

def train_model(params, train, valid, hvalid, outcome='ilive'):
    params.update({
                   'objective':'binary:logistic',
                   'eval_metric':'auc',
                   'seed':9999,
                   'tree_method':'hist',
                   'grow_policy':'lossguide'
                  })
    params['max_leaves'] = int(params['max_leaves'])
    params['max_bin'] = int(params['max_bin'])
    dtrain = xgboost.DMatrix(train.drop([outcome], axis=1), train[outcome])
    dvalid = xgboost.DMatrix(valid.drop([outcome], axis=1), valid[outcome])
    del train, valid
    model = xgboost.train(params, dtrain, 2048, evals=[(dvalid, 'eval')], early_stopping_rounds=16)
    dtest = xgboost.DMatrix(hvalid.drop([outcome], axis=1), hvalid[outcome])
    y_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    auroc = sklearn.metrics.roc_auc_score(hvalid[outcome], y_pred)
    return 1-auroc

def choose_model(train, valid, hvalid, spec_name):
    best = fmin(fn=partial(train_model,
                           train=train,
                           valid=valid,
                           hvalid=hvalid),
                space=space,
                algo=tpe.suggest,
                max_evals=256,
                rstate=np.random.RandomState(9999))
    best['max_leaves'] = 128 - best['max_leaves']
    best['max_bin'] = 2 + best['max_bin']
    best['subsample'] = 1.5 - best['subsample']
    best['colsample_bytree'] = 1.5 - best['colsample_bytree']
    best['colsample_bylevel'] = 1.5 - best['colsample_bylevel']
    with open(spec_name + ' Hyperparameters.p', 'wb') as file:
        pickle.dump(best, file, protocol=pickle.HIGHEST_PROTOCOL)
    return best

df_2016_train = pickle.load(open('Full 2016 Training Set.p', 'rb'))
df_2016_valid = pickle.load(open('Full 2016 Validation Set.p', 'rb'))
df_2016_hvalid = pickle.load(open('Full 2016 Hypervalidation Set.p', 'rb'))
df_2016_NICU_train = pickle.load(open('NICU 2016 Training Set.p', 'rb'))
df_2016_NICU_valid = pickle.load(open('NICU 2016 Validation Set.p', 'rb'))
df_2016_NICU_hvalid = pickle.load(open('NICU 2016 Hypervalidation Set.p', 'rb'))
df_varlist = pd.read_csv('Neonatal Mortality Predictor List.csv')
prenatal_vars = df_varlist[df_varlist['prenatal']==1]['feature'].tolist()
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

best_prenatal = choose_model(df_2016_train[prenatal_vars],
                              df_2016_valid[prenatal_vars],
                              df_2016_hvalid[prenatal_vars],
                              'Prenatal')

best_NICU = choose_model(df_2016_NICU_train[prenatal_vars],
                         df_2016_NICU_valid[prenatal_vars],
                         df_2016_NICU_hvalid[prenatal_vars],
                         'NICU')

best_post_birth = choose_model(df_2016_train,
                               df_2016_valid,
                               df_2016_hvalid,
                               'Postnatal')
    
print('Run time: ' + str(time() - start_time) + ' seconds')