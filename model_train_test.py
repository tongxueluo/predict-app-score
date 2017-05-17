__author__ = 'luoyuan'
'''This py file used for training the final data.
   It uses xgboost classifier for training.
   It reports both the gini for the test data. Also the gini for the cross validation of training data.'./report.
   It can all give the report for the information value of each feature.
   It can return or plot the feature importance in the training model.
   It also import rank_ordering.py for generating a standard rank_order report.
   The results saved to folder './report'. '''

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import numpy as np
import sklearn.metrics
import os
import time
from datetime import datetime
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
target = 'Bad_flag_worst6'
from rank_ordering import rank_order

output_path = './report'
if not os.path.exists(output_path):
    os.mkdir(output_path)

def get_xgb_feat_importances(clf):
    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.booster().get_fscore()
    else:
        # clf is an instance of xgb.Booster.
        fscore = clf.get_fscore()
    feat_importances = []
    for ft, score in fscore.iteritems():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)
    return feat_importances

def xgmodelfit(alg, df_train, df_test, useTrainCV=True, cv_folds=5, return_gini=False,
               early_stopping_rounds=50, plot_import=True, return_feat_importance=False):
    '''this function used for creating XGBoost models and perform cross-validation for the training data df_train,
    then test on test data df_test. Note that the input data should be in DataFrame form'''
    features = [f for f in df_train.columns if f not in ['Unnamed: 0',target]]
    f = open(output_path+'/train_test_report_%d_%s.txt' % (len(features), datetime.now().strftime('%m%d%H%M') ),'w+')
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(df_train[features].values,  label=df_train[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        n=cvresult.shape[0]
        alg.set_params(n_estimators=n)
        gini_cv_train = 2*cvresult['test-auc-mean'].loc[n-1]-1
        print 'Gini (CV train): %f'% gini_cv_train
        print >> f, 'Gini (CV train): %f'% gini_cv_train
    #Fit the algorithm on the data
    alg.fit(df_train[features], df_train[target], eval_metric='auc')
    dtest_predprob = alg.predict_proba(df_test[features])[:,1]
    gini_test = 2*sklearn.metrics.roc_auc_score(df_test[target], dtest_predprob)-1
    print "Gini (Test): %f" % gini_test
    print >> f, 'Gini (CV train): %f'% gini_test
    print >> f, 'Model details: %s'% alg
    print >> f, '%s features used'% len(features)

    '''generate rank_ordering report'''
    df_test_copy = df_test.copy()
    df_test_copy['predprob'] = dtest_predprob
    rank = rank_order()
    rank.rank_order(df_test_copy, y_col=target, x_col='predprob', ascending = False, version=len(features))

    #the feature importance vasulization
    if plot_import:
        plot_importance(alg, importance_type='gain', max_num_features=60)
        plt.show()
    if return_gini:
        if return_feat_importance:
            feat_importances = get_xgb_feat_importances(alg)
            return gini_cv_train, gini_test, feat_importances
        else:
            return gini_cv_train, gini_test
    elif return_feat_importance:
        feat_importances = get_xgb_feat_importances(alg)
        return feat_importances

def xgradient_boost(df_train, df_test, return_feat_importance=False, plot_import=True, return_gini=False):
    xgb1 = XGBClassifier(n_estimators=100)
    if return_gini or return_feat_importance:
        return xgmodelfit(xgb1, df_train, df_test,return_gini=return_gini, plot_import=plot_import,
                                                                   return_feat_importance=return_feat_importance)
    else:
        xgmodelfit(xgb1, df_train, df_test, plot_import=plot_import)

def WOE(good_acounts, bad_acounts, g_total, b_total, x=0.5):
    pg = (good_acounts+0.5)/g_total
    gb = (bad_acounts+0.5)/b_total
    return np.log(pg/gb)

def iv_caculator(df_data, col_ca = 'categories'):
    uniques = sorted([i for i in df_data[col_ca].unique() if not np.isnan(i)])
    uniques = np.array(uniques).astype(int)
    iv_sum = 0
    bad_total_counts = df_data[target].sum()
    good_total_counts = len(df_data)-df_data[target].sum()
    for category in uniques:
        data = df_data[df_data[col_ca]==category]
        total_counts = len(data)
        bad_counts = data[target].sum()
        good_counts = total_counts - bad_counts
        bad_ratio = bad_counts/bad_total_counts
        good_ratio = good_counts/good_total_counts
        iv = (good_ratio-bad_ratio)*WOE(good_counts, bad_counts, good_total_counts, bad_total_counts)
        iv_sum += iv
    return iv_sum

def iv(df_data,col, bin_no = 5):
    df_copy = df_data.copy()
    uniques = sorted([i for i in df_data[col].unique() if not np.isnan(i)])
    length = len(uniques)
    min_ = min(uniques)
    max_ = max(uniques)
    labels_ = range(bin_no)
    step_ = 1
    if length>bin_no:
        max_min = max_ - min_
        gap = float(max_min)/bin_no
        bins = [gap*i+min_ for i in xrange(bin_no+1)]
        df_copy['categories'] = pd.cut(df_copy[col], bins, labels=labels_)
        return iv_caculator(df_copy)
    elif length>0:
        labels_ = range(length-1)
        bins = uniques
        df_copy['categories'] = pd.cut(df_copy[col], bins, labels=labels_)
        return iv_caculator(df_copy)
    else:
        return 0.

'''this function get infomation value for all features'''
def get_information_value(df_train, N=66, return_feature_names=True):
    features = [f for f in df_train.columns if f not in ['Unnamed: 0', target]]
    ivs = []
    for col in features:
        ivs.append(iv(df_train, col))
    features_info_val = dict(zip(features,ivs))
    features_info_val = pd.DataFrame.from_dict(features_info_val.items())
    features_info_val.sort_values([1],inplace=True,ascending=False)
    features_info_val.to_csv(output_path+'/features_IV%s.csv'%(datetime.now().strftime('%H%M')))
    if return_feature_names:
        features_N = features_info_val[0].values[0:N]
        return features_N

class model:
    def __init__(self, input_path, plot_import = False, N_selection = 50):
        self.input_path = input_path
        self.filelist = os.listdir(self.input_path)[1:]
        self.plot_import = plot_import
        self.N = N_selection
    def train(self, drops_back = True):
        df_train = pd.read_csv(self.input_path+'/'+'train_data.csv')
        df_test = pd.read_csv(self.input_path+'/'+'test_data.csv')

        if drops_back:  # by experiment it shows that droping these features can improve gini by around 0.01.
            drops_back_to_v1 = ['perc_unsecured_loantypes','total_rateofinterests','total_paymentfrequencies','paymentfrequency_ratios','total_actualpaymentamounts',
                            'mostfreq_acct_types', 'ave_high_credit_amts','ave_enq_amts','status_type']
            drops = drops_back_to_v1
            df_train.drop(df_train[drops],axis=1, inplace=True)
            df_test.drop(df_test[drops], axis=1, inplace=True)

        print "--------The results for using %s features: ----------"%(len(df_train.columns)-2)
        xgradient_boost(df_train, df_test,plot_import=self.plot_import)

        features_N = get_information_value(df_train, N=self.N)
        features_N = [i for i in features_N] + [target]
        print "--------The results after feature selection by information value. Using %s features: ----------"%(len(features_N)-1)
        xgradient_boost(df_train[features_N], df_test[features_N],plot_import=self.plot_import)

if __name__ == '__main__':
    start_time = time.time()
    input_path = './final_features'
    classifer = model(input_path)
    classifer.train()
    print '-----running time: %s----------'%(time.time()-start_time)








