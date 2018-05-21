import os
import pickle
import pandas as pd
import numpy as np
from time import time

from sklearn.model_selection import train_test_split
import sklearn.metrics
import matplotlib.pyplot as plt
import xgboost

start_time = time()

os.chdir(os.path.dirname(__file__))

def train_final_model(params, X, y):
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
    return model

def plot_metrics(model, y_test, y_pred, model_name):
    my_dpi = 96
    xgboost.plot_importance(model,
                            max_num_features=16,
                            importance_type='weight',
                            show_values=False,
                            xlabel='Number of Splits').figure.savefig(model_name + '_weight_VI.png', dpi=my_dpi)
    xgboost.plot_importance(model,
                            max_num_features=16,
                            importance_type='gain',
                            show_values=False,
                            xlabel='Average Gini Gain per Split').figure.savefig(model_name + '_gain_VI.png', dpi=my_dpi)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    auroc = sklearn.metrics.roc_auc_score(y_test, y_pred)
    aupr = sklearn.metrics.average_precision_score(y_test, y_pred)
    logloss = sklearn.metrics.log_loss(y_test, y_pred)
    brier = sklearn.metrics.brier_score_loss(y_test, y_pred)
    plt.figure(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)
    plt.step(fpr, tpr, color='b', alpha=0.2,
             where='post')
    plt.fill_between(fpr, tpr, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve for ' + model_name + ': AUC = {0:0.4f}'.format(auroc))
    plt.savefig(model_name + '_ROC.png', dpi=my_dpi)
    
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_pred)
    plt.figure(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)
    plt.step(precision, recall, color='b', alpha=0.2,
             where='post')
    plt.fill_between(precision, recall, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve for ' + model_name + ': AUC = {0:0.4f}'.format(aupr))
    plt.savefig(model_name + '_PR.png', dpi=my_dpi)
    return {'model_name':model_name, 'auroc':auroc, 'aupr':aupr, 'logloss':logloss, 'brier':brier}

df_2015 = pickle.load(open("df_2015.p", "rb"))
df_2016 = pickle.load(open("df_2016.p", "rb"))
df_varlist = pd.read_csv('Neonatal Mortality Predictor List.csv')
pre_birth_vars = df_varlist[df_varlist['pre_birth']==1]['feature'].tolist()
del df_varlist

best_NICU = pickle.load(open("NICU_best_params.p", "rb"))
NICU_model = train_final_model(best_NICU,
                         df_2016[df_2016['ab_nicu']==2].drop(['ilive', 'ab_nicu'], axis=1),
                         df_2016[df_2016['ab_nicu']==2]['ilive'])
plot_metrics(NICU_model,
             df_2015[df_2015['ab_nicu']==2]['ilive'],
             NICU_model.predict(xgboost.DMatrix(df_2015[df_2015['ab_nicu']==2].drop(['ilive', 'ab_nicu'], axis=1),
                                                df_2015[df_2015['ab_nicu']==2]['ilive']),
                                ntree_limit=NICU_model.best_ntree_limit),
             'NICU')
del NICU_model

best_pre_birth = pickle.load(open("pre_birth_best_params.p", "rb"))
pre_birth_model = train_final_model(best_pre_birth,
                              df_2016[pre_birth_vars],
                              df_2016['ilive'])
pre_birth_pred = pre_birth_model.predict(xgboost.DMatrix(df_2016[pre_birth_vars],
                                                          df_2016['ilive']),
                                         ntree_limit=pre_birth_model.best_ntree_limit)
pre_birth_qtiles = np.percentile(pre_birth_pred, np.linspace(0, 100, 10000))
with open('pre_birth_qtiles.p', 'wb') as file:
    pickle.dump(pre_birth_qtiles, file, protocol=pickle.HIGHEST_PROTOCOL)
del pre_birth_pred
plot_metrics(pre_birth_model,
             df_2015['ilive'],
             pre_birth_model.predict(xgboost.DMatrix(df_2015[pre_birth_vars],
                                                     df_2015['ilive']),
                                     ntree_limit=pre_birth_model.best_ntree_limit),
             'pre_birth')
with open('pre_birth_model.p', 'wb') as file:
    pickle.dump(pre_birth_model, file, protocol=pickle.HIGHEST_PROTOCOL)
del pre_birth_model

best_post_birth = pickle.load(open("post_birth_best_params.p", "rb"))
post_birth_model = train_final_model(best_post_birth,
                               df_2016.drop(['ilive', 'ab_nicu'], axis=1),
                               df_2016['ilive'])
del df_2016
y_test = df_2015['ilive']
d_X = xgboost.DMatrix(df_2015.drop(['ilive', 'ab_nicu'], axis=1),
                      y_test)
del df_2015
y_pred = post_birth_model.predict(d_X,
                                  ntree_limit=post_birth_model.best_ntree_limit)
del d_X
plot_metrics(post_birth_model,
             y_test,
             y_pred,
             'post_birth')
del post_birth_model
    
print('Run time: ' + str(time() - start_time) + ' seconds')