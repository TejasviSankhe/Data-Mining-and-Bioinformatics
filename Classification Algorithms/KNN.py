
# coding: utf-8
"""
Description: Implement KNN Classification for a given dataset 
Group Members: Mitali Bhiwande | Sumedh Ambokar | Tejasvi Sankhe
"""
# In[1]:

import numpy as np
import random
from scipy.spatial import distance
import operator
import math
import sys

# In[2]:

#function to convert input file data to float value
def is_float(string):
    try:
        val = float(string)
        return True
    except ValueError:
        return False


# In[3]:

#Function to normalize data using mean and standard deviation
def normalize_data(mean, std_dev, data, labels):
    for i in range(len(data)):
        normalised=((data[i]-mean)/std_dev).tolist()
        normalised.append(labels[i])
        data[i]=normalised
    return data


# In[11]:

# Fucntions to find training and testing datasets  and run knn on each fold to compute performace measures
def generate_datasets(test_fold_number, records, fold_size):
    starti = (test_fold_number - 1) * fold_size
    endi = test_fold_number * fold_size - 1
    copy = list(records)
    testing = copy[starti : endi+1]
    del copy[starti : endi]
    return copy, testing

def run_kfold_validation(k_folds, records,k):
    k_rows = math.ceil(len(records)/k_folds)
    acc=[]
    pre=[]
    recl=[]
    fm=[]
    for i in range(1,k_folds+1):
        training_data, testing_data = generate_datasets(i, records, k_rows)
        accuracy, precision, recall , fmeasure=process_knn(training_data,testing_data,k)
        acc.append(accuracy)
        pre.append(precision)
        recl.append(recall)
        fm.append(fmeasure)
    print("ACCURACY: ", np.mean(acc) *100.0, "%")
    print("PRECISION: ", np.mean(pre))
    print("RECALL: ", np.mean(recl))
    print("F-MEASURE:  ",np.mean(fmeasure))


# In[12]:

#derives the labels for test instances based on nearest neighbor labels and voting
def get_label(nearest_neighbors):
    
    labels=[row[len(row)-1] for row in nearest_neighbors]
    dic={}
    for i in range(len(labels)):
        if labels[i] in dic:
            dic[labels[i]]=dic[labels[i]]+1
        else:
            dic[labels[i]]=1
    highest_value=(max(dic.values()))
   
    label=[(k) for k, v in dic.items() if v == highest_value]
    return label


# In[13]:

#computes euclidean distance as similarity measure and finds the top K nearest neighbors
def compute_knn(normalized_data, test_data, mean, std_dev, k):
    dist=[]
    normalized_test_data=(test_data[0:len(test_data)-1]-mean)/std_dev
    for i in range(len(normalized_data)):
        dist.append((normalized_data[i],(distance.euclidean(normalized_data[i][0:len(normalized_data[0])-1],normalized_test_data))))
    dist.sort(key=operator.itemgetter(1))
    nearest_neighbors=[]
    for i in range(k):
        nearest_neighbors.append(dist[i][0])
    knn_label=get_label(nearest_neighbors)
    return knn_label


# In[14]:

# Evaluates the various performance measures
def performance_measure(test_labels, predicted_labels):
    tp=0.0
    tn=0.0
    fp=0.0
    fn=0.0
    for i in range(len(test_labels)):
        if(test_labels[i]== 1.0 and predicted_labels[i][-1]==1.0):
            tp=tp+1
        elif(test_labels[i]== 0.0 and predicted_labels[i][-1]==1.0):
            fp=fp+1
        elif(test_labels[i]== 0.0 and predicted_labels[i][-1]==0.0):
            tn=tn+1
        else:
            fn=fn+1
    accuracy = (tp+tn)/len(test_labels) 
    precision = tp/(tp+fp) 
    recall = tp/(tp+fn) 
    F_measure = (2*recall*precision) / (recall+precision)
    
    return (accuracy, precision, recall, F_measure)


# In[15]:

# REads data from file and makes it ready for processing.
def fetch_data(filename):
    original_data=[]
    with open(filename) as textFile:
        lines = [line.replace("\n","").split("\t") for line in textFile]
        
        for line in lines:
            original_data.append([float(f) if is_float(f) else f for f in line])
        for i in range(len(original_data[0])):
            if isinstance(original_data[0][i],str):
                unique_val=set([line[i] for line in original_data])
                dic = dict()
                counter = 0
                for val in unique_val:
                    if val in dic:
                        continue
                    dic[val] = counter
                    counter=counter+1
                for line in original_data:
                    line[i]=dic.get(line[i])
    return original_data   


# In[16]:

# Computes the predicted label set and performance measures.
def process_knn(train_data,test_data,k):
    
    data_set=[line[0:-1] for line in train_data]
    train_labels=[line[-1] for line in train_data]
    test_labels=[line[-1] for line in test_data]
    predicted_labels=[]
    
    mean=np.mean(data_set, axis=0)
    std_dev=np.std(data_set,axis=0)
    
    normalized_data=normalize_data(mean, std_dev, data_set, train_labels)
    
    for x in range(len(test_data)):
        knn_label=compute_knn(normalized_data ,test_data[x],mean,std_dev,k)
        predicted_labels.append(knn_label)
        
    accuracy, precision, recall , fmeasure= performance_measure(test_labels, predicted_labels)
    return accuracy, precision, recall , fmeasure


# In[20]:

if len(sys.argv) > 1:

    filePath = sys.argv[1]
    number_of_fold = int(sys.argv[2])
    K_value= int(sys.argv[3])
    original_data=fetch_data(filePath)
    run_kfold_validation(number_of_fold,original_data,K_value)
else:
    print("Not sufficient Inputs")


'''
#if no separate test and train
original_data=fetch_data("project3_dataset2.txt")
run_kfold_validation(10,original_data,5)
'''

# In[24]:
'''

#if separate test and train
train_data=fetch_data("project3_dataset3_train.txt")
test_data=fetch_data("project3_dataset3_test.txt")
for i in range(1,30):
    print(i)
    accuracy, precision, recall , fmeasure=process_knn(train_data,test_data,i)
    print("ACCURACY: ", accuracy *100.0, "%")
    print("PRECISION: ", precision)
    print("RECALL: ", recall)
    print("F-MEASURE:  ", fmeasure)

'''