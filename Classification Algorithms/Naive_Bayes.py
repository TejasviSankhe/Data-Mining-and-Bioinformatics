import sys
import numpy as np
import scipy.stats as sc

# reads data from file, converts numerical data to float and determines if categorical data is present
def readData(filepath):
    dataset=[]
    numerical=set()
    categorical=set()
    data_file = open(filepath, 'r')
    raw_data = [line.split('\t') for line in data_file.readlines()]
    
    row=raw_data[0]
    temp=[]
    for i in range(0,len(row)):
        try:
            temp.append(float(row[i]))
            numerical.add(i)
        except ValueError:
            temp.append(row[i])
            categorical.add(i)
    dataset.append(temp)          
    numerical.remove(len(dataset[0])-1)
    #print (numerical,categorical)
    return raw_data,numerical,categorical

# partitions dataset in training and testing dataset based on k value and number of iterations
def KFold(dataset,k,iteration):
    
    starti=(iteration-1) * (k)
    endi=(iteration * k) - 1
    copy=list(dataset)
    testing= copy[starti:endi+1]
    del copy[starti:endi]
    training=copy
    
    return training,testing
    
# implementation of naive bayes algorithm with K fold validation for large test samples
def NaiveBayesValidation(train_samples, test_samples, numerical, categorical):
    
    predictedLabel=[]
    samples_0=list()
    samples_1=list()
    sample_1_num=[]
    sample_0_num=[]
    sample_1_cat=[]
    sample_0_cat=[]
    
    total_1=0
    total_0=0
    
    # seperate training dataset instances by class value
    classIndex=len(train_samples[0])-1
    
    for i in range(0, train_samples.shape[0]):
        if int(train_samples[i][classIndex])==1 : 
            temp_num=[]
            temp_cat=[]
            total_1 +=1
            for n in numerical:
                temp_num.append(float(train_samples[i][n]))
            for n in categorical:
                temp_cat.append(train_samples[i][n])
            sample_1_num.append(temp_num)
            sample_1_cat.append(temp_cat)
        else:
            temp_num=[]
            temp_cat=[]
            total_0 +=1
            for n in numerical:
                temp_num.append(float(train_samples[i][n]))
            
            for n in categorical:
                temp_cat.append(train_samples[i][n])
            sample_0_num.append(temp_num)
            sample_0_cat.append(temp_cat)
    
    if len(numerical) > 0:
        meanS_1=np.array(sample_1_num).mean(axis=0)
        stdS_1=np.array(sample_1_num).std(axis=0)
        meanS_0=np.array(sample_0_num).mean(axis=0)
        stdS_0=np.array(sample_0_num).std(axis=0)
        
    if(len(categorical) > 0):
            trans_data1 = [list(x) for x in zip(*sample_1_cat)]
            trans_data0 = [list(x) for x in zip(*sample_0_cat)]
    
    
    for i in range(0,len(test_samples)):
        test_query=test_samples[i]
        test_query_num=[]
        test_query_cat=[]
        for i in range (0,len(test_query)-1):
            if i in numerical:
                test_query_num.append(float(test_query[i]))
            elif i in categorical:
                test_query_cat.append(test_query[i])
        
        probability_0 = 1.0
        probability_1 = 1.0  

        # calculate mean and std dev for numerical attributes based on class value
        if len(numerical) > 0:
            for i in range(0,len(test_query_num)):
                probability_0 *= sc.norm(meanS_0[i], stdS_0[i]).pdf(test_query_num[i])
                probability_1 *= sc.norm(meanS_1[i], stdS_1[i]).pdf(test_query_num[i])

        if(len(categorical) > 0):
            for i in range(0,len(test_query_cat)):
                probability_1 *= (trans_data1[i].count(test_query_cat[i])/total_1)
                probability_0 *= (trans_data0[i].count(test_query_cat[i])/total_0)

        probability_0 *= total_0 / float(total_1+total_0)
        probability_1 *= total_1 / float(total_1+total_0)

        # Check for which label the probibility is highest and add that label in the predicted_labels
        if probability_0 > probability_1:
            predictedLabel.append(0)
        else:
            predictedLabel.append(1)
            
    #print(predictedLabel)       
    return predictedLabel


# Calculate the performance measures for the predicted values based on orignal data
def PerformanceMeasure(predicted_labels , orignal_labels):

    tp=0.0
    tn=0.0
    fp=0.0
    fn=0.0
    
    #orignal labels has the test dataset, last column contains the orignal label
    num=len(predicted_labels)
    for i in range (0,num):
        if(int(predicted_labels[i])==1 and int(orignal_labels[i][-1])==1):
            tp+=1
        elif(int(predicted_labels[i])==1 and int(orignal_labels[i][-1])==0):
            fp+=1
        elif(int(predicted_labels[i])==0 and int(orignal_labels[i][-1])==0):
            tn+=1
        else:
            fn+=1
            
    accuracy = (tp+tn)/num
    try:
        precision = tp/(tp+fp)
    except ZeroDivisionError:
        precision=0
    try:
        recall = tp/(tp+fn)
    except ZeroDivisionError:
        recall=0
    try:
        F_measure = (2*recall*precision) / (recall+precision)
    except ZeroDivisionError:
        F_measure=0
        
    return (accuracy, precision, recall, F_measure)
            
# function for calling test samples with validation function
def validation():
    global filepath
    data, numerical,categorical = readData(filepath)
    k_fold=10  # 10 fold validation
    k_rows = int(len(data)/k_fold)

    #print(len(data), k)
    total_accuracy,total_precision,total_recall,total_Fm = 0,0,0,0

    for i in range(1,k_fold+1):
        train, test = KFold(data,k_rows,i)
        predicted = NaiveBayesValidation(np.array(train),np.array(test),numerical,categorical)
        accuracy, precision, recall, F_measure= PerformanceMeasure(predicted, test)

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_Fm += F_measure

    print("Performance measures for ", filepath)
    print("Average accuracy :", total_accuracy/k_fold)
    print("Average precision:", total_precision/k_fold)
    print("Average recall   :", total_recall/k_fold)
    print("Average F measure:", total_Fm/k_fold)
    
#validation()

# implementation of naive bayes algorithm, predicting a single test query
def NaiveBayes(train_samples, test_query, numerical, categorical):
    
    samples_0=list()
    samples_1=list()
    sample_1_num=[]
    sample_0_num=[]
    sample_1_cat=[]
    sample_0_cat=[]
    
    test_query_num=[]
    test_query_cat=[]
    total_1=0
    total_0=0
    
    for i in range (0,len(test_query)):
        if i in numerical:
            test_query_num.append(test_query[i])
        elif i in categorical:
            test_query_cat.append(test_query[i])
    
    # seperate training dataset instances by class value
    classIndex=len(train_samples[0])-1
    
    for i in range(0, train_samples.shape[0]):
        if int(train_samples[i][classIndex])==1 : 
            temp_num=[]
            temp_cat=[]
            total_1 +=1
            for n in numerical:
                temp_num.append(float(train_samples[i][n]))
            for n in categorical:
                temp_cat.append(train_samples[i][n])
            sample_1_num.append(temp_num)
            sample_1_cat.append(temp_cat)
        else:
            temp_num=[]
            temp_cat=[]
            total_0 +=1
            for n in numerical:
                temp_num.append(float(train_samples[i][n]))
            for n in categorical:
                temp_cat.append(train_samples[i][n])
            sample_0_num.append(temp_num)
            sample_0_cat.append(temp_cat)
            
    probability_0 = 1.0
    probability_1 = 1.0  
    px=1.0
    
    # calculate mean and std dev for numerical attributes based on class value
    if len(numerical) > 0:
        
        meanS_1=np.array(sample_1_num).mean(axis=0)
        stdS_1=np.array(sample_1_num).std(axis=0)
        meanS_0=np.array(sample_0_num).mean(axis=0)
        stdS_0=np.array(sample_0_num).std(axis=0) 
        #print(meanS_0, stdS_0, "and", meanS_1, stdS_1) 
        for i in range(0,len(test_query_num)):
            probability_0 *= sc.norm(meanS_0[i], stdS_0[i]).pdf(test_query_num[i])
            probability_1 *= sc.norm(meanS_1[i], stdS_1[i]).pdf(test_query_num[i])
    
    # count frequency for categorical values       
    if(len(categorical) > 0):
        trans_data1 = [list(x) for x in zip(*sample_1_cat)]
        trans_data0 = [list(x) for x in zip(*sample_0_cat)]

        for i in range(0,len(test_query_cat)):
            probability_1 *= (trans_data1[i].count(test_query_cat[i])/total_1)
            probability_0 *= (trans_data0[i].count(test_query_cat[i])/total_0)
            px *= (trans_data1[i].count(test_query_cat[i])+trans_data0[i].count(test_query_cat[i]))/len(train_samples)
    
    # multiply the categorical probabilities with numerical
    probability_0 *= total_0 / float(total_1+total_0)
    probability_0 /= px
    probability_1 *= total_1 / float(total_1+total_0)
    probability_1 /= px

    #print(probability_0,"and", probability_1)
    return probability_0,probability_1


# Method for predicting class for single query
def query(queryip):
    global filepath
    data,numerical,categorical = readData(filepath)
    #query=['sunny','cool','high','weak']
    h0,h1=NaiveBayes(np.array(data), queryip, numerical, categorical)
    print("Query:" ,queryip)
    print("P(H0|X):",h0,"\nP(H1|X):", h1)
    if h0 > h1:
        print("Predicted Label: 0")
    else:
        print("Predicted Label: 1")


def main():

    global filepath
    filepath = sys.argv[1]
    choice=int(input("Enter your choice\n1 Predict a single query \n2 Perform 10 fold validation\n"))

    if(choice==1) :
        #call function to predict a class for query, change file path in function implementation if required
        queryip = input("Enter the query seperated by |: ").split('|')
        query(queryip)
    else:
        #call function to perform 10-fold validation on the dataset, change file path in function implementation if required
        validation()

if __name__ == "__main__":
    main()



