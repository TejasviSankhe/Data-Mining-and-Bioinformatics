########## Run Boosting ###########

import sys
from Boosting_Class import Boosting

# Function used to fetch the data from a file
def fetch_data(filename):
    original_data=[]
    with open(filename) as textFile:
        lines = [line.replace("\n","").split("\t") for line in textFile]
        for line in lines:
            original_data.append([val for val in line])
    return original_data

# Function used to split the dataset into training and testing parts based on passed start index
def generate_datasets(test_fold_number, records, fold_size):

    starti = (test_fold_number - 1) * fold_size
    endi = test_fold_number * fold_size
    copy = list(records)
    testing = copy[starti : endi]
    del copy[starti : endi]
    return copy, testing

	
# Function to perform K Folds cross validations	
def run_kfold_validation(k_folds, records, number_of_learners, bag_size_percent):
    
    acc, prec, recl, F_mea = [], [], [], []
    k_rows = int(len(records)/k_folds)
    for i in range(1,k_folds+1):
        training_data, testing_data = generate_datasets(i, records, k_rows)
        boosting = Boosting(training_data, bag_size_percent, number_of_learners)
        accuracy, precision, recall, F_measure = boosting.calculate_measures(testing_data, '1', '0')
        print('iteration --> ', i)
        print('Accuracy: ', accuracy*100, '%')
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F_measure: ', F_measure)
        acc.append(accuracy)
        prec.append(precision)
        recl.append(recall)
        F_mea.append(F_measure)
    print(' ##### Cummulative Output ##### ')
    print('Accuracy: ', sum(acc) / float(len(acc)) * 100, '%')
    print('Precision: ', sum(prec) / float(len(prec)))
    print('Recall: ', sum(recl) / float(len(recl)))
    print('F_measure: ', sum(F_mea) / float(len(F_mea)))

	
file_name =  input("Enter File Name: ")
number_of_folds = input("Enter Number of Folds: ")
number_of_learners = input("Enter Number of Decision Tree Learners: ")
bag_size_percent = input("Enter bag size to be considered in percentage: ")

run_kfold_validation(int(number_of_folds), fetch_data(file_name), int(number_of_learners), int(bag_size_percent))