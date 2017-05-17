
'''this py file get features from (preprocessed) demographic data data_70&30.
Main process here is convert categorical variables into dummy/indicator variables.
Then take all the numerical features.
The features (together with the targets) saved to file './demogh_data'''

import pandas as pd
import numpy as np
import os

def dummy_replacement(pddata, list):
    for column in list:
        numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if pddata[column].dtype not in numerics and len(set(pddata[column].values))<10:
            s = pd.get_dummies(pddata[column],prefix=column)
            pddata = pd.concat([pddata, s], axis=1)
            pddata.__delitem__(column)
    return pddata

class get_demogh_features:
    def __init__(self, input_path):
        self.input_path = input_path
        self.filelist = os.listdir(input_path)

    def __data_preprocessing(self):
        print '---------------do some data_preprocessing first--------------------'
        for file in self.filelist:
            if 'data_70' in file:
                demogh_70 = pd.read_csv(self.input_path+'/'+file, low_memory=False)
            elif 'data_30' in file:
                demogh_30 = pd.read_csv(self.input_path+'/'+file, low_memory=False)

        dummy_list = ['card_name','aip_status','status_type','reject_reason_code',
                      'reject_reason_desc','intl_trn','fee_code','override_fee_code',
                      'acq_source','se_code','lead_code','mktg_code','app_gender','app_pan',
                      'mob_verified','email','marital_status','edu_qualification','app_res_city',
                      'app_res_pincode','res_type','permanent_same','employment_type',
                      'industry_type','designation','office_city','id_proof','existing_bank','app_has_card',
                      'existing_card_issuer','app_title','app_state','app_existing_other_loan_cc',
                      'app_existing_rbl_cust','app_international_transaction_fl','app_si_ecs_flg',
                      'app_resident_india','app_perm_state','app_perm_city','app_employment_type',
                      'app_pref_mailing_addr','app_account_type','app_fcu_check_status']
        self.demogh_train = dummy_replacement(demogh_70, dummy_list)
        self.demogh_test = dummy_replacement(demogh_30, dummy_list)
        print '---------------data_preprocessing done--------------------'

    def get_features(self):
        self.__data_preprocessing()

        print '---------------getting features--------------------'
        self.train_apprefno_list = self.demogh_train['apprefno'].values
        self.test_apprefno_list = self.demogh_test['apprefno'].values

        print "number of data_70 and 30: ", len(self.train_apprefno_list), len(self.test_apprefno_list)
        numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        demogh_70_numeric = self.demogh_train.select_dtypes(include=numerics)
        demogh_30_numeric = self.demogh_test.select_dtypes(include=numerics)

        feature_names = []
        for name in demogh_70_numeric.columns:
            if 'dt' not in name and 'time' not in name and 'worst_dpd6' not in name and 'bad_flag_worst6' not in name:
                feature_names.append(name)
        feature_names = feature_names[1:]
        print 'len(feature_names): ', len(feature_names)


        demogh_features_70, demogh_features_30 = [],[]
        targets_70, targets_30 = [], []
        for i in xrange(len(demogh_70_numeric)):
            f = demogh_70_numeric.iloc[i][feature_names].values
            demogh_features_70.append(f)
            targets_70.append(self.demogh_train.iloc[i]['bad_flag_worst6'])
        for i in xrange(len(demogh_30_numeric)):
            f = demogh_30_numeric.iloc[i][feature_names].values
            demogh_features_30.append(f)
            targets_30.append(self.demogh_test.iloc[i]['bad_flag_worst6'])
        demogh_features_70, demogh_features_30 = np.array(demogh_features_70), np.array(demogh_features_30)
        targets_70, targets_30 = np.array(targets_70), np.array(targets_30)

        output_folder = './demogh_data'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if np.isinf(demogh_features_70).any() or np.isnan(demogh_features_70).any() \
            or np.isinf(demogh_features_30).any() or np.isnan(demogh_features_30).any():
            print "There's NaN values! Check!"

        np.save(output_folder+'/demogh_features_70.npy', demogh_features_70)
        np.save(output_folder+'/demogh_features_30.npy', demogh_features_30)
        np.save(output_folder+'/targets_70.npy', targets_70)
        np.save(output_folder+'/targets_30.npy', targets_30)
        np.savetxt(output_folder+'/demogh_feature_names.txt', feature_names, delimiter=" ", fmt="%s")
        print '---------------features (together with the targets) saved.--------------------'
if __name__ == '__main__':
    import time
    start_time = time.time()
    input_path = './RBL_70_30_processed'
    get_demogh_features = get_demogh_features(input_path)
    get_demogh_features.get_features()
    print '-----running time: %s----------'%(time.time()-start_time)
