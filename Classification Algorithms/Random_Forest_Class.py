############## Random Forest Class #################

import itertools
import random
from Decision_Tree_Class import Decision_Tree

class Random_Forest:
    
    def __init__(self, records, number_of_trees, attributes_percent):
        
        # This variable stores all the Decision Tree in the forest in a form of a list
        self.forest = []
        attributes_set = self.get_attribute_list((len(records[0]) - 1), attributes_percent)
        for num in range(number_of_trees):
            # Get the bootstrapped sample from the training dataset
            bagged_records = self.get_bagged_records(records)
            decision_tree = Decision_Tree(bagged_records, attributes_set, True)
            self.forest.append(decision_tree)

    # Function used to generate unique sets features of specific size
    def get_attribute_list(self, total_attributes, attributes_percent):
        
        number_of_attributes = round(total_attributes*attributes_percent/100)
        all_attribute_sets = list(itertools.combinations(tuple(range(total_attributes)), number_of_attributes))
        return all_attribute_sets
    
    # Function to perform Bootstrap sampling on the training dataset
    def get_bagged_records(self, records):
        
        indices_list = [random.choice(range(len(records))) for _ in range(len(records))]
        bagged_records = []
        for index in indices_list:
            bagged_records.append(records[index])
        return bagged_records
    
    # Function to traverse all the Decision trees and perform a class prediction using their cummulative value
    def predict_label(self, record):
        
        labels = []
        label_dict = {}
        max_count = 0
        label_pred = None
        for decision_tree in self.forest:
            labels.append(decision_tree.predict_label(record))
        for label in labels:
            if label in label_dict:
                label_dict[label] = label_dict[label] + 1
            else:
                label_dict[label] = 1
            if max_count < label_dict[label]:
                max_count = label_dict[label]
                label_pred = label
        return label_pred
        
	# Function to calculate performance measures
    def calculate_measures(self, test_records, true_label, false_label):

        true_pos, true_neg, false_pos, false_neg = 0.0, 0.0, 0.0, 0.0
        predicted_labels = []
        for record in test_records:
            predicted_labels.append(self.predict_label(record[0:-1]))
        total_records=len(test_records)
        for i in range (0,total_records):
            if(predicted_labels[i]==true_label and test_records[i][-1]==true_label):
                true_pos+=1
            elif(predicted_labels[i]==true_label and test_records[i][-1]==false_label):
                false_pos+=1
            elif(predicted_labels[i]==false_label and test_records[i][-1]==false_label):
                true_neg+=1
            else:
                false_neg+=1
        accuracy = (true_pos+true_neg)/total_records
        precision = 0 if true_pos == 0 and false_pos == 0 else true_pos/(true_pos+false_pos)
        recall = 0 if true_pos == 0 and false_neg == 0 else true_pos/(true_pos+false_neg)
        F_measure = 0 if recall == 0 and precision == 0 else (2*recall*precision) / (recall+precision)

        return accuracy, precision, recall, F_measure
