################################  K-NEAREST NEIGHBOR CLASSIFICATION ############################################

To run the code just pass the data file path as an input command line parameter
- Below is a sample execution command:

>> python KNN.py project3_dataset1.txt 10 5

Here, 
INPUT: 

project3_dataset1.txt = filePath 
10 = number_of_fold
5 = K_value for KNN

OUTPUT:
- The code will output the average performace measures over the input number of folds such as accuracy, precision, recall and f-measure for the provided input parameters.

#################################  NAIVE BAYES ALGORITHM  ##############################################

- To run the code excute the pyton script and pass the file name as command line argument. Sample command is as follows:
> python Naive_Bayes.py project3_dataset2.txt
> python Naive_Bayes.py project3_dataset3.txt

The window will prompt with a choice, select accordingly.
Enter your choice
1 Predict a single query
2 Perform 10 fold validation

If you choose '1', a further prompt will appear to input a query seperated by '|'. Enter the query accordingly.

Example execution command as follows:
Enter the query seperated by |: sunny|cool|high|weak


Input:
- File path as command line argument.
- The choice for query prediction or 10-fold validation.
- In case of query prediction, the test query attributes seperated by '|' 

Output:

- For query prediction, Class probabilities i.e. H0 and H1, and the predicted class the query belongs to.
- For 10 fold validation, name of dataset used, average accuracy, average precision, average recall, average F-measure.


#################################  Decision Tree CART  ##############################################

Pre-requisite:
1. Decision_Tree_Class.py needs to be present in the folder for implementing the algorithm

Executable File: Decision_Tree.py

Steps:
1. Run the file using python command
	>> python Decision_Tree.py
2. The program will ask for a file name to perform the algorithm on.
3. After providing the file name program asks for operations:
	>> Print --> to print the decision tree
	>> Validate --> to perform k-folds cross validation
4. If Validate is selected program will ask for size of K to implement K-folds

Output:
1. For Operation Print the program will print the decision tree structure on the comman prompt
2. For Operation Validate the program will output Accuracy, Precision, Recall and F_Measure for every K-fold iteration and finally output the cummulative results.


#################################   Random Forest  ##############################################


Pre-requisite:
1. Decision_Tree_Class.py and Random_Forest_Class.py needs to be present in the folder for implementing the algorithm

Executable File: Random_Forest.py

Steps:
1. Run the file using python command
	>> python Random_Forest.py
2. The program will ask for a file name to perform the algorithm on.
3. The program will ask for number of folds
4. The program will ask for number of trees to be created
5. The program will ask for percentage of features which needs to be considered for this execution.

Output:
1. The program will output Accuracy, Precision, Recall and F_Measure for every K-fold iteration and finally output the cummulative results.


#################################   Boosting   ##############################################


Pre-requisite:
1. Decision_Tree_Class.py and Boosting_Class.py needs to be present in the folder for implementing the algorithm

Executable File: Boosting.py

Steps:
1. Run the file using python command
	>> python Boosting.py
2. The program will ask for a file name to perform the algorithm on.
3. The program will ask for number of folds
4. The program will ask for number of decision tree learners to be created
5. The program will ask for percentage of the training set which needs to be used as the sample.

Output:
1. The program will output Accuracy, Precision, Recall and F_Measure for every K-fold iteration and finally output the cummulative results.