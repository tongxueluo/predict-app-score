# predict-app-score
(step_1) preprocess.py for preprocessing all raw data. 

(step_2) get_account_equiry_features.py and get_demogh_features.py for feature engineering preprocessed historical and demographic data, respectively. Two sets of features generated. 

(step_2.5) combine_features_for_training.py for combining these two sets to get final features for training, 
generating train and test data saved to './train_test_data'. 

(step_3) model_train_test.py for training model on train_data and testing on test_data saved in the './train_test_data' folder. Model reports generated. 

Input: ./raw_data_folder.  
Output after this data_processing-modeling pipeline: './report' folder with gini, rank ordering and Information Value reports.
