__author__ = 'luoyuan'
'''the preprocess.py load and preprocess the raw data
Perform functions like 1) replace nan values and missing data,
                       2) convert date-month-year to numbers for further feature engineering
                       3) convert certain categorical variables into dummy/indicator variables for our interest
                       4) Can save the 'bad' data (i.e. 'Bad_flag_worst6 = 1') as separate csv files and have a look,
                        see any pattern found.
Save proprecessed data to './RBL_70_30_processed' file.
'''

import pandas as pd
import os
import numpy as np

def replace_nan(pddata, columns):
    #get most frequnt value in the column
    for column in columns:
        total_len = len(pddata[column])
        nonnan_len = pddata[column].count()
        portion =  float(nonnan_len)/total_len
        most_frequent_mode = pddata[column].mode()
        if len(most_frequent_mode)>0:
            most_frequent_mode = most_frequent_mode[0]
        else:
            most_frequent_mode = 0
        if portion>=0.5 or 'dt' in column or 'time' in column:
            pddata[column].fillna(most_frequent_mode, inplace=True)
        elif portion<0.5:
            pddata[column].fillna(0, inplace=True)   #if the data is scarce, just fill 0 to the NaN
    return pddata

def convert_dmy(pddata, columns):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' ]
    mon_numbers = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    mon_number_dict = dict(zip(months, mon_numbers))
    for column in columns:
        if 'dt' in column or 'time' in column:
            for i in xrange(len(pddata[column])):
                day, mon, year = pddata[column][i].split('-')
                day = int(day)
                mon = mon_number_dict[mon]
                year = int(year)*365
                count = day+mon+year
                pddata.set_value(i, column, count)
    return pddata

def dummy_replacement(pddata, *args):
    for column in args:
        s = pd.Series(list(pddata[column])).astype(int)
        pddata = pd.concat([pddata, pd.get_dummies(s,prefix=column)], axis=1)
    return pddata

class preprocess_data:
    def __init__(self, input_path):
        self.input_path = input_path
        self.file_list = os.listdir(self.input_path)

    def preprocess(self):
        print '---------------start preprocessing--------------------'
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

        self.train_apprefno_list = list(set(demogh_70['apprefno']))
        self.test_apprefno_list = list(set(demogh_30['apprefno']))
        print "number of data_70 and 30: ", len(self.train_apprefno_list), len(self.test_apprefno_list)


        # first step: fill all NaN with the value defined in function replace_nan
        demogh_70 = replace_nan(demogh_70, demogh_70.columns)
        demogh_30 = replace_nan(demogh_30, demogh_30.columns)
        account_70 = replace_nan(account_70, account_70.columns)
        account_30 = replace_nan(account_30, account_30.columns)
        enquiry_70 = replace_nan(enquiry_70, enquiry_70.columns)
        enquiry_30 = replace_nan(enquiry_30, enquiry_30.columns)


        # second step: convert all day-month-year data to numbers, prepared for further feature engineering
        demogh_70 = convert_dmy(demogh_70, demogh_70.columns)
        demogh_30 = convert_dmy(demogh_30, demogh_30.columns)
        account_70 = convert_dmy(account_70, account_70.columns)
        account_30 = convert_dmy(account_30, account_30.columns)
        enquiry_70 = convert_dmy(enquiry_70, enquiry_70.columns)
        enquiry_30 = convert_dmy(enquiry_30, enquiry_30.columns)

        # thrid step: get the dummy representation for our interested features
        account_70 = dummy_replacement(account_70, 'acct_type', 'writtenoffandsettled')
        account_30 = dummy_replacement(account_30, 'acct_type', 'writtenoffandsettled')
        enquiry_70 = dummy_replacement(enquiry_70,'enq_purpose')
        enquiry_30 = dummy_replacement(enquiry_30,'enq_purpose')

        output_file = './RBL_70_30_processed'
        if not os.path.exists(output_file):
           os.makedirs(output_file)
        demogh_70.to_csv(output_file+'/'+'data_70_processed.csv')
        demogh_30.to_csv(output_file+'/'+'data_30_processed.csv')
        account_70.to_csv(output_file+'/'+'account_70_processed.csv')
        account_30.to_csv(output_file+'/'+'account_30_processed.csv')
        enquiry_70.to_csv(output_file+'/'+'enquiry_70_processed.csv')
        enquiry_30.to_csv(output_file+'/'+'enquiry_30_processed.csv')

        print '---------------end of preprocessing. preprocessed data saved-------------------'

    def save_bad_data(self):
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
        bad_demogh_70 = demogh_70.loc[demogh_70['bad_flag_worst6']==1]
        bad_demogh_30 = demogh_30.loc[demogh_30['bad_flag_worst6']==1]
        self.bad_apprefno_list_70 = list(set(bad_demogh_70['apprefno']))
        self.bad_apprefno_list_30 = list(set(bad_demogh_30['apprefno']))
        print "number of bad_data_70 and 30: ", len(self.bad_apprefno_list_70), len(self.bad_apprefno_list_30)


        bad_account_70 = account_70[account_70['apprefno'].isin(self.bad_apprefno_list_70)]
        bad_account_30 = account_30[account_30['apprefno'].isin(self.bad_apprefno_list_30)]
        bad_enquiry_70 = enquiry_70[enquiry_70['apprefno'].isin(self.bad_apprefno_list_70)]
        bad_enquiry_30 = enquiry_30[enquiry_30['apprefno'].isin(self.bad_apprefno_list_30)]

        output_file = './RBL_70_30_bad'
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        bad_demogh_70.to_csv(output_file+'/bad_data_70.csv')
        bad_demogh_30.to_csv(output_file+'/bad_data_30.csv')
        bad_account_70.to_csv(output_file+'/bad_account_70.csv')
        bad_account_30.to_csv(output_file+'/bad_account_30.csv')
        bad_enquiry_70.to_csv(output_file+'/bad_enquiry_70.csv')
        bad_enquiry_30.to_csv(output_file+'/bad_enquiry_30.csv')
        print '---------------"bad" data saved-----------------------'

if __name__ == '__main__':
    import time
    start_time = time.time()
    input_path = './RBL_70_30'
    preprocess = preprocess_data(input_path=input_path)
    preprocess.preprocess()
    preprocess.save_bad_data()
    print '------------running time %s seconds---------'%(time.time()-start_time)



