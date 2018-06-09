import os
import pickle
import pandas as pd
import numpy as np
from time import time
import sklearn.metrics
from scipy.stats import norm

import xgboost

start_time = time()

os.chdir(os.path.dirname(__file__))

def train_final_model(params, train, valid, outcome='ilive'):
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
    return model

def format_ci(metric, metric_pm):
    return '{:.5f}'.format(metric) + '±' + '{:.5f}'.format(metric_pm)

def compute_intervals(y_test, y_pred, model_name):
    n_obs = len(y_test)
    alpha = .05
    norm_coef = norm.ppf(1-(alpha/2))
    auroc = sklearn.metrics.roc_auc_score(y_test, y_pred)
    auroc_pm = norm_coef*(auroc*(1-auroc)/n_obs)**0.5
    auroc_ci = format_ci(auroc, auroc_pm)
    aupr = sklearn.metrics.average_precision_score(y_test, y_pred)
    aupr_pm = norm_coef*(aupr*(1-aupr)/n_obs)**0.5
    aupr_ci = format_ci(aupr, aupr_pm)
    logloss = sklearn.metrics.log_loss(y_test, y_pred) 
    logloss_pm = norm_coef*np.std(np.add(np.multiply(y_test, np.log(y_pred)), np.multiply(np.subtract(1,y_test), np.log(np.subtract(1,y_pred)))))/(n_obs**0.5)
    logloss_ci = format_ci(logloss, logloss_pm)
    brier = sklearn.metrics.brier_score_loss(y_test, y_pred)
    brier_pm = norm_coef*np.std(np.subtract(y_test, y_pred)**2)/(n_obs**0.5)
    brier_ci = format_ci(brier, brier_pm)
    return {'model_name':model_name, 'auroc':auroc_ci, 'aupr':aupr_ci, 'logloss':logloss_ci, 'brier':brier_ci}

def get_metrics(model_name, train, valid, test, outcome='ilive'):
    best_model_params = pickle.load(open(model_name + ' Hyperparameters.p', 'rb'))
    best_model = train_final_model(best_model_params,
                                   train,
                                   valid,
                                   outcome=outcome)
    pred = best_model.predict(xgboost.DMatrix(train.drop([outcome], axis=1), train[outcome]),
                              ntree_limit=best_model.best_ntree_limit)
    qtiles = np.percentile(pred, np.linspace(0, 100, 10000))
    with open(model_name + ' Quantiles.p', 'wb') as file:
        pickle.dump(qtiles, file, protocol=pickle.HIGHEST_PROTOCOL)
    del pred
    metrics_dict = compute_intervals(test[outcome],
                                     best_model.predict(xgboost.DMatrix(test.drop([outcome], axis=1), test[outcome]),
                                                        ntree_limit=best_model.best_ntree_limit),
                                     model_name)
    feature_gains = best_model.get_score(importance_type='gain')
    gain_sum = sum(feature_gains.values())
    feature_weights = best_model.get_score(importance_type='weight')
    weight_sum = sum(feature_weights.values())
    top_5_features = sorted(feature_gains, key=feature_gains.get, reverse=True)[0:5]
    top_5_gains = [feature_gains[k]/gain_sum for k in top_5_features]
    top_5_weights = [feature_weights[k]/weight_sum for k in top_5_features]
    feature_data = top_5_features + top_5_gains + top_5_weights
    feature_data_headers = ['Feature {} Name'.format(i) for i in range(1, 6)] + \
                           ['Feature {} Gain'.format(i) for i in range(1, 6)] + \
                           ['Feature {} Weight'.format(i) for i in range(1, 6)]
    metrics_dict.update(dict(zip(feature_data_headers, feature_data)))
    with open(model_name + ' Model.p', 'wb') as file:
        pickle.dump(best_model, file, protocol=pickle.HIGHEST_PROTOCOL)
    return metrics_dict

pd.set_option('display.max_columns', None)
pd.set_option('precision', 5)
        
df_2016_train = pickle.load(open('Full 2016 Training Set.p', 'rb'))
df_2016_valid = pickle.load(open('Full 2016 Validation Set.p', 'rb'))
df_2015 = pickle.load(open('Full 2015.p', 'rb'))
results_table = pd.DataFrame()

results_table = results_table.append(get_metrics('Postnatal',
                                                 df_2016_train,
                                                 df_2016_valid,
                                                 df_2015),
                                     ignore_index=True)

df_varlist = pd.read_csv('Neonatal Mortality Predictor List.csv')
prenatal_vars = df_varlist[df_varlist['prenatal']==1]['feature'].tolist()
del df_varlist
df_2016_train = df_2016_train[prenatal_vars]
df_2016_valid = df_2016_valid[prenatal_vars]
df_2015 = df_2015[prenatal_vars]
results_table = results_table.append(get_metrics('Prenatal',
                                                 df_2016_train,
                                                 df_2016_valid,
                                                 df_2015),
                                     ignore_index=True)
del df_2016_train, df_2016_valid, df_2015, prenatal_vars

df_2016_NICU_train = pickle.load(open('NICU 2016 Training Set.p', 'rb'))
df_2016_NICU_valid = pickle.load(open('NICU 2016 Validation Set.p', 'rb'))
df_2015_NICU = pickle.load(open('NICU 2015.p', 'rb'))
results_table = results_table.append(get_metrics('NICU',
                                                 df_2016_NICU_train,
                                                 df_2016_NICU_valid,
                                                 df_2015_NICU),
                                     ignore_index=True)
del df_2016_NICU_train, df_2016_NICU_valid, df_2015_NICU

print(results_table)
    
print('Run time: ' + str(time() - start_time) + ' seconds')