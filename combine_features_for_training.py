
'''this py file just combine the account_equiry features and demogh features for the training.
Final features saved to file './final_features'. '''

import numpy as np
import pandas as pd
import os

class get_final_features:
    def __init__(self, input_path1, input_path2):
        self.input_path1 = input_path1
        self.input_path2 = input_path2
        self.filelist1 = os.listdir(self.input_path1)
        self.filelist2 = os.listdir(self.input_path2)
    def combine_save_features(self):
        print '-----------start combining--------------'
        for file in self.filelist1:
            if 'features_70' in file:
                x_train1 = np.load(self.input_path1+'/'+file)
            elif 'features_30' in file:
                x_test1 = np.load(self.input_path1+'/'+file)
            elif 'targets_70' in file:
                y_train1 = np.load(self.input_path1+'/'+file)
            elif 'targets_30' in file:
                y_test1 = np.load(self.input_path1+'/'+file)
            elif 'feature_names' in file:
                feature_names1 = open(self.input_path1+'/'+file)
                feature_names1 = feature_names1.read().split('\n')[:-1]
                print 'len(features1): ',len(feature_names1)
        for file in self.filelist2:
            if 'features_70' in file:
                x_train2 = np.load(self.input_path2+'/'+file)
            elif 'features_30' in file:
                x_test2 = np.load(self.input_path2+'/'+file)
            elif 'targets_70' in file:
                y_train2 = np.load(self.input_path2+'/'+file)
            elif 'targets_30' in file:
                y_test2 = np.load(self.input_path2+'/'+file)
            elif 'feature_names' in file:
                feature_names2 = open(self.input_path2+'/'+file)
                feature_names2 = feature_names2.read().split('\n')[:-1]
                print 'len(features2): ',len(feature_names2)

        print 'x_train1.shape=%s, x_train2.shape=%s: '%(x_train1.shape, x_train2.shape)
        print 'x_test1.shape=%s, x_test2.shape=%s'%(x_test1.shape, x_test2.shape)

        x_train = np.concatenate(tuple([x_train1, x_train2]),axis=1)
        y_train = y_train1
        x_test = np.concatenate(tuple([x_test1, x_test2]),axis=1)
        y_test = y_test1
        feature_names = feature_names1 + feature_names2

        print 'x_train.shape=%s, y_train.shape=%s'%(x_train.shape,y_train.shape)
        print 'x_test.shape%s, y_test.shape=%s'%(x_test.shape,y_test.shape)
        print 'len(features): ',len(feature_names)

        output_file = './train_test_data'
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        np.save(output_file+'/x_train.npy', x_train)
        np.save(output_file+'/x_test.npy', x_test)
        np.save(output_file+'/y_train.npy', y_train)
        np.save(output_file+'/y_test.npy', y_test)
        np.savetxt(output_file+'/feature_names.txt', feature_names, delimiter=" ", fmt="%s")

        #all save the data as cvs
        columes = feature_names + ['Bad_flag_worst6']
        values_train =  np.concatenate(tuple([x_train,y_train.reshape((y_train.shape[0],1))]),axis=1)
        data_train = pd.DataFrame(values_train, columns=columes)
        values_test =  np.concatenate(tuple([x_test,y_test.reshape((y_test.shape[0],1))]),axis=1)
        data_test = pd.DataFrame(values_test, columns=columes)
        data_train.to_csv(output_file+'/train_data.csv')
        data_test.to_csv(output_file+'/test_data.csv')
        print '-----data saved-----------'

if __name__ == '__main__':
    import time
    start_time = time.time()
    input_path1 = './accout_equiry_data'
    input_path2 = './demogh_data'
    get_final_features = get_final_features(input_path1, input_path2)
    get_final_features.combine_save_features()
    print '-----running time: %s----------'%(time.time()-start_time)
