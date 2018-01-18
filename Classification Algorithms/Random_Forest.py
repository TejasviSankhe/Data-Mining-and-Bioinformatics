########## Run Random Forest ###########

import sys
from Random_Forest_Class import Random_Forest

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
def run_kfold_validation(k_folds, records, number_of_trees, attribute_percent):
    
    acc, prec, recl, F_mea = [], [], [], []
    k_rows = int(len(records)/k_folds)
    for i in range(1,k_folds+1):
        
        training_data, testing_data = generate_datasets(i, records, k_rows)
        random_forest = Random_Forest(training_data, number_of_trees, attribute_percent)
        accuracy, precision, recall, F_measure = random_forest.calculate_measures(testing_data, '1', '0')
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
	
# Input module created for user's ease
file_name = input("Enter File Name: ")
number_of_folds = input("Enter Number of Folds: ")
number_of_trees = input("Enter Number of Tree: ")
attribute_percent = input("Enter percent of features to be considered: ")

run_kfold_validation(int(number_of_folds), fetch_data(file_name), int(number_of_trees), int(attribute_percent))