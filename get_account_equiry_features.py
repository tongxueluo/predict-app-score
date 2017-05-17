
'''this py file construct features from (preprocessed) account_ and equiry_ data,
the meaning of the features can be understood as their names,
as well as the functions that get them.
The features (together with the targets) saved to file './accout_equiry_data'''

import os
import numpy as np
import pandas as pd

def get_payment_history_avg_dpd_0_29_bucket(accouts, col1='paymenthistory1',col2='paymenthistory2'):
    seq1 = accouts[col1].values
    seq2 = accouts[col2].values
    counts = len(seq1)
    history_nums =  []
    history_length = 0
    for i in xrange(counts):
        history_num1 = [int(s) for s in seq1[i] if s.isdigit()]
        len1 = len(history_num1)
        history_length += len1
        if len1>=3:
            history_nums.append(history_num1)
        history_num2 = [int(s) for s in seq2 if s.isdigit()]
        len2 = len(history_num2)
        history_length += len2
        if len2>=3:
            history_nums.append(history_num2)
    history_nums = [i for nums in history_nums
                      for i in nums]
    dpd_total = 0
    for i in xrange(len(history_nums)-2):
        if i%3 == 0:
            dpd = int(history_nums[i])*100 + int(history_nums[i+1])*10 + int(history_nums[i+2])
            if dpd<=29:
                dpd_total += dpd
    if counts>0:
        avg_dpd = float(dpd_total)/counts
        mean_len = history_length/counts/2
    else:
        avg_dpd = float(dpd_total)/1.
        mean_len = history_length/2.
    return avg_dpd, mean_len,

def get_diff_lastpaymt_opened_dt(accounts, col1='opened_dt', col2='last_paymt_dt'):
    length = len(accounts)
    total = sum(accounts[col2].values-accounts[col1].values)
    if length>0:
        mean = total/length
    else:
        mean = total/1.
    return total, mean

def get_min_months_last_30_plus(accouts, col1='paymenthistory1',col2='paymenthistory2'):
    seq1 = accouts[col1].values
    seq2 = accouts[col2].values
    length = len(seq1)
    mon = 50
    mons = []
    for i in xrange(length):
        mon1 = 0
        mon2 = 0
        history_num1 = [int(s) for s in seq1[i] if s.isdigit()]
        if len(history_num1)>=3:
           for k in xrange(len(history_num1)-2):
                if k%3 == 0:
                    dpd = int(history_num1[k])*100 + int(history_num1[k+1])*10 + int(history_num1[k+2])
                    if dpd>29:
                        mon1 = k/3+1
        history_num2 = [int(s) for s in seq2 if s.isdigit()]
        if len(history_num2)>=3:
           for k in xrange(len(history_num2)-2):
                if k%3 == 0:
                    dpd = int(history_num2[k])*100 + int(history_num2[k+1])*10 + int(history_num2[k+2])
                    if dpd>29:
                        mon2 = k/3+1
        mons.append(max(mon1, mon2))
    if max(mons)>0:
        mon = 18-max(mons)
    return mon

def get_balance_credit_cash_features(accounts, col1 = 'cur_balance_amt', col2 = 'creditlimit', col3 = 'cashlimit'):
    total_cur_bal_amt = accounts[col1].sum()
    mean_cur_bal_amt = accounts[col1].mean()
    total_creditlimit = accounts[col2].sum()
    mean_creditlimit = accounts[col2].mean()
    mean_cashlimit = accounts[col3].mean()

    if (mean_creditlimit+ mean_cashlimit)==0.:   #prevent inf number by simple assumption
        return 1., 1.
    ratio_means = float(mean_cur_bal_amt)/(mean_creditlimit+ mean_cashlimit)
    if total_creditlimit>0:
        ratio_currbalance_creditlimit = total_cur_bal_amt/total_creditlimit
    else:
        ratio_currbalance_creditlimit = 1.
    if ratio_means>0:
        utilisation_trend = ratio_currbalance_creditlimit/ratio_means
    else:
        utilisation_trend = 1.
    return ratio_currbalance_creditlimit, utilisation_trend

def get_count_enquiry_recency(equiries):
    return len(equiries)

def get_diff_open_enquiry_dt(equiries,col1='dt_opened',col2='enquiry_dt'):
    counts = len(equiries)
    total = sum(equiries[col1].values-equiries[col2].values)
    if counts>0:
        mean = total/counts
    else:
        mean = total/1.
    return total, mean
def get_equiry_purpose_feature(equiries, list):
    enq_purpose = []
    for i in list:
        col = 'enq_purpose_%d'%i
        try:
            enq_purpose.append(equiries[col].sum())
        except:
            enq_purpose.append(0)
    return enq_purpose

def get_paymentamount_features(accouts, col1 = 'actualpaymentamount', col2='paymentfrequency'):
    total_amount = accouts[col1].sum()
    total_pay_frequency = accouts[col2].sum()
    payment_freq_ratio = 0.
    if total_pay_frequency>0:
        payment_freq_ratio = float(total_amount)/total_pay_frequency
    return total_amount, total_pay_frequency, payment_freq_ratio

def get_total_rateofinterests(accouts, col = 'rateofinterest'):
    seq = [float(s) for s in accouts[col].values if s.isdigit()]
    return sum(seq)

def get_mostfreq_acct_type(accouts, col = 'acct_type'):
    modes = accouts[col].mode()
    if len(modes)>0:
        return accouts[col].mode()[0]
    else:
        return 0

def get_ave_high_credit_amt(accouts, col ='high_credit_amt'):
    return accouts[col].mean()

def get_ave_enq_amt(equiries, col = 'enq_amt'):
    return equiries[col].mean()

def get_perc_unsecured_loantypes(accouts, col = 'acct_type'):
    unsecured_types = [5, 6, 8, 9, 10, 12,16,35,40, 41, 43, 80, 81, 88, 89, 90,91,92,93,94,95, 96, 97, 98, 99, 0]
    counts = 0
    total_counts = len(accouts)
    vals = accouts[col].values
    for val in vals:
        if val in unsecured_types:
            counts += 1
    if total_counts>0:
        return float(counts)/total_counts
    else:
        return 0.

def get_targets(data, col='bad_flag_worst6'):
    return data[col].values

class feature_engineering:
    def __init__(self, input_path):
        self.input_path = input_path
        self.file_list = os.listdir(self.input_path)
    def get_features(self):
        print '---------------start feature engineering--------------------'
        print '......will take some time, please be patient......'
        for file in self.file_list:
            if 'data_70' in file:
                demogh_70 = pd.read_csv(self.input_path+'/'+file, low_memory=False)
            elif 'data_30' in file:
                demogh_30 = pd.read_csv(self.input_path+'/'+file, low_memory=False)
            elif 'account_70' in file:
                account_70 = pd.read_csv(self.input_path+'/'+file, low_memory=False)
            elif 'account_30' in file:
                account_30 = pd.read_csv(self.input_path+'/'+file, low_memory=False)
            elif 'enquiry_70' in file:
                enquiry_70 = pd.read_csv(self.input_path+'/'+file, low_memory=False)
            elif 'enquiry_30' in file:
                enquiry_30 = pd.read_csv(self.input_path+'/'+file, low_memory=False)
        self.train_apprefno_list = demogh_70['apprefno'].values
        self.test_apprefno_list = demogh_30['apprefno'].values
        print "number of data_70 and 30: ", len(self.train_apprefno_list), len(self.test_apprefno_list)

        '''define the features'''
        train_total_diff_lastpaymt_opened_dt = []
        test_total_diff_lastpaymt_opened_dt = []
        train_mean_diff_lastpaymt_opened_dt = []
        test_mean_diff_lastpaymt_opened_dt = []
        train_payment_history_avg_dpd_0_29_bucket = []
        test_payment_history_avg_dpd_0_29_bucket = []
        train_min_months_last_30_plus = []
        test_min_months_last_30_plus = []
        train_ratio_currbalance_creditlimit = []
        test_ratio_currbalance_creditlimit = []
        train_utilisation_trend = []
        test_utilisation_trend = []
        train_count_enquiry = []
        test_count_enquiry = []
        train_mean_diff_open_enquiry_dt = []
        test_mean_diff_open_enquiry_dt = []
        train_payment_history_mean_length = []
        test_payment_history_mean_length = []
        train_enquiry_purposes = []
        test_enquiry_purposes = []
        '''update features'''
        train_total_actualpaymentamounts = []
        test_total_actualpaymentamounts = []
        train_total_paymentfrequencies = []
        test_total_paymentfrequencies = []
        train_paymentfrequency_ratios = []
        test_paymentfrequency_ratios = []
        train_total_rateofinterests = []
        test_total_rateofinterests = []
        train_mostfreq_acct_types = []
        test_mostfreq_acct_types = []
        train_ave_high_credit_amts = []
        test_ave_high_credit_amts = []
        train_ave_enq_amts = []
        test_ave_enq_amts = []
        train_perc_unsecured_loantypes = []
        test_perc_unsecured_loantypes = []

        list_purposes = list(set(enquiry_70['enq_purpose'].values))
        train_targets = []
        test_targets = []
        for apprefno in self.train_apprefno_list:
            t_d, m_d = get_diff_lastpaymt_opened_dt(account_70[account_70['apprefno']==apprefno])
            train_total_diff_lastpaymt_opened_dt.append(t_d)
            train_mean_diff_lastpaymt_opened_dt.append(m_d)

            avg_dpd, h_l = get_payment_history_avg_dpd_0_29_bucket(account_70[account_70['apprefno']==apprefno])
            train_payment_history_avg_dpd_0_29_bucket.append(avg_dpd)
            train_payment_history_mean_length.append(h_l)

            m_l_3 = get_min_months_last_30_plus(account_70[account_70['apprefno']==apprefno])
            train_min_months_last_30_plus.append(m_l_3)

            r_c_c, u_t = get_balance_credit_cash_features(account_70[account_70['apprefno']==apprefno])
            train_ratio_currbalance_creditlimit.append(r_c_c)
            train_utilisation_trend.append(u_t)

            c_enq = get_count_enquiry_recency(enquiry_70[enquiry_70['apprefno']==apprefno])
            train_count_enquiry.append(c_enq)

            to_o_e, m_o_e = get_diff_open_enquiry_dt(enquiry_70[enquiry_70['apprefno']==apprefno])
            train_mean_diff_open_enquiry_dt.append(m_o_e)

            #get equiry purpose feature matrix. note there're 59 different types of purpose
            enq_p = get_equiry_purpose_feature(enquiry_70[enquiry_70['apprefno']==apprefno],list_purposes)
            train_enquiry_purposes.append(enq_p)

            '''updates'''
            t_p_a, freq, ratio = get_paymentamount_features(account_70[account_70['apprefno']==apprefno])
            train_total_actualpaymentamounts.append(t_p_a)
            train_total_paymentfrequencies.append(freq)
            train_paymentfrequency_ratios.append(ratio)

            roi = get_total_rateofinterests(account_70[account_70['apprefno']==apprefno])
            train_total_rateofinterests.append(roi)

            m_f_t = get_mostfreq_acct_type(account_70[account_70['apprefno']==apprefno])
            train_mostfreq_acct_types.append(m_f_t)

            a_h_c = get_ave_high_credit_amt(account_70[account_70['apprefno']==apprefno])
            train_ave_high_credit_amts.append(a_h_c)

            e_a = get_ave_enq_amt(enquiry_70[enquiry_70['apprefno']==apprefno])
            train_ave_enq_amts.append(e_a)

            p_u_l = get_perc_unsecured_loantypes(account_70[account_70['apprefno']==apprefno])
            train_perc_unsecured_loantypes.append(p_u_l)

            tg = get_targets(demogh_70[demogh_70['apprefno']==apprefno])
            train_targets.append(tg)

        for apprefno in self.test_apprefno_list:
            t_d, m_d = get_diff_lastpaymt_opened_dt(account_30[account_30['apprefno']==apprefno])
            test_total_diff_lastpaymt_opened_dt.append(t_d)
            test_mean_diff_lastpaymt_opened_dt.append(m_d)

            avg_dpd, h_l = get_payment_history_avg_dpd_0_29_bucket(account_30[account_30['apprefno']==apprefno])
            test_payment_history_avg_dpd_0_29_bucket.append(avg_dpd)
            test_payment_history_mean_length.append(h_l)

            m_l_3 = get_min_months_last_30_plus(account_30[account_30['apprefno']==apprefno])
            test_min_months_last_30_plus.append(m_l_3)

            r_c_c, u_t = get_balance_credit_cash_features(account_30[account_30['apprefno']==apprefno])
            test_ratio_currbalance_creditlimit.append(r_c_c)
            test_utilisation_trend.append(u_t)

            c_enq = get_count_enquiry_recency(enquiry_30[enquiry_30['apprefno']==apprefno])
            test_count_enquiry.append(c_enq)

            to_o_e, m_o_e = get_diff_open_enquiry_dt(enquiry_30[enquiry_30['apprefno']==apprefno])
            test_mean_diff_open_enquiry_dt.append(m_o_e)

            #get equiry purpose feature matrix. note there're 59 different types of purpose
            enq_p = get_equiry_purpose_feature(enquiry_30[enquiry_30['apprefno']==apprefno],list_purposes)
            test_enquiry_purposes.append(enq_p)

            '''updates'''
            t_p_a, freq, ratio = get_paymentamount_features(account_30[account_30['apprefno']==apprefno])
            test_total_actualpaymentamounts.append(t_p_a)
            test_total_paymentfrequencies.append(freq)
            test_paymentfrequency_ratios.append(ratio)

            roi = get_total_rateofinterests(account_30[account_30['apprefno']==apprefno])
            test_total_rateofinterests.append(roi)

            m_f_t = get_mostfreq_acct_type(account_30[account_30['apprefno']==apprefno])
            test_mostfreq_acct_types.append(m_f_t)

            a_h_c = get_ave_high_credit_amt(account_30[account_30['apprefno']==apprefno])
            test_ave_high_credit_amts.append(a_h_c)

            e_a = get_ave_enq_amt(enquiry_30[enquiry_30['apprefno']==apprefno])
            test_ave_enq_amts.append(e_a)

            p_u_l = get_perc_unsecured_loantypes(account_30[account_30['apprefno']==apprefno])
            test_perc_unsecured_loantypes.append(p_u_l)

            tg = get_targets(demogh_30[demogh_30['apprefno']==apprefno])
            test_targets.append(tg)
        train_targets = np.array(train_targets).ravel()
        test_targets = np.array(test_targets).ravel()

        train_features_tuple = [train_total_diff_lastpaymt_opened_dt, train_mean_diff_lastpaymt_opened_dt,
                                   train_payment_history_avg_dpd_0_29_bucket, train_min_months_last_30_plus,
                                   train_ratio_currbalance_creditlimit, train_utilisation_trend,
                                   train_count_enquiry, train_mean_diff_open_enquiry_dt,
                                   train_payment_history_mean_length,
                                   train_total_actualpaymentamounts, train_total_paymentfrequencies,
                                   train_paymentfrequency_ratios, train_total_rateofinterests, train_mostfreq_acct_types,
                                   train_ave_high_credit_amts,train_ave_enq_amts, train_perc_unsecured_loantypes,
                               train_enquiry_purposes,]
        test_features_tuple = [test_total_diff_lastpaymt_opened_dt, test_mean_diff_lastpaymt_opened_dt,
                                   test_payment_history_avg_dpd_0_29_bucket, test_min_months_last_30_plus,
                                   test_ratio_currbalance_creditlimit, test_utilisation_trend,
                                   test_count_enquiry, test_mean_diff_open_enquiry_dt,
                                   test_payment_history_mean_length,
                                   test_total_actualpaymentamounts, test_total_paymentfrequencies,
                                   test_paymentfrequency_ratios, test_total_rateofinterests, test_mostfreq_acct_types,
                                   test_ave_high_credit_amts,test_ave_enq_amts, test_perc_unsecured_loantypes,
                               test_enquiry_purposes,]


        feature_names = ['total_diff_lastpaymt_opened_dt', 'mean_diff_lastpaymt_opened_dt',
                                   'payment_history_avg_dpd_0_29_bucket', 'min_months_last_30_plus',
                                   'ratio_currbalance_creditlimit', 'utilisation_trend',
                                   'count_enquiry', 'mean_diff_open_enquiry_dt',
                                   'payment_history_mean_length','total_actualpaymentamounts',
                                   'total_paymentfrequencies', 'paymentfrequency_ratios', 'total_rateofinterests',
                                    'mostfreq_acct_types','ave_high_credit_amts', 'ave_enq_amts', 'perc_unsecured_loantypes']\
                        + ['enq_purpose_%d'%int(i) for i in list_purposes]

        axis0_train = train_targets.shape[0]
        axis0_test = test_targets.shape[0]
        train_features = []
        test_features = []
        for item in train_features_tuple:
            item = np.array(item)
            if len(item.shape)<2:
                item = np.array(item).reshape(axis0_train,1)
                train_features.append(item)
            else:
                train_features.append(item)
        for item in test_features_tuple:
            item = np.array(item)
            if len(item.shape)<2:
                item = np.array(item).reshape(axis0_test,1)
                test_features.append(item)
            else:
                test_features.append(item)
        train_features = np.concatenate(tuple(train_features), axis=1)
        test_features = np.concatenate(tuple(test_features), axis=1)
        print 'feature shapes: ', train_features.shape, test_features.shape
        print 'len(feature_names): ', len(feature_names)

        '''save the features'''
        output_folder = './accout_equiry_data'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_list = os.listdir(output_folder)[1:]
        version_no = len(file_list)/5+1
        np.save(output_folder+'/accout_equiry_features_70_v%d.npy'%version_no, train_features)
        np.save(output_folder+'/accout_equiry_features_30_v%d.npy'%version_no, test_features)
        np.save(output_folder+'/accout_equiry_targets_70_v%d.npy'%version_no, train_targets)
        np.save(output_folder+'/accout_equiry_targets_30_v%d.npy'%version_no, test_targets)
        np.savetxt(output_folder+'/accout_equiry_feature_names_v%d.txt'%version_no, feature_names, delimiter=" ", fmt="%s")
        print '---------------end of feature engineering. features (together with the targets) saved.--------------------'

if __name__ == '__main__':
    import time
    start_time = time.time()
    input_path = './RBL_70_30_processed'
    feature_engineering = feature_engineering(input_path)
    feature_engineering.get_features()
    print '-----running time: %s----------'%(time.time()-start_time)
