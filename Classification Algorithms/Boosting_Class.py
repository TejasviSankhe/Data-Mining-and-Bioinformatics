############ Boosting Class ###################

import random
from Decision_Tree_Class import Decision_Tree

class Boosting:
    
    def __init__(self, records, training_set_percent, number_of_learners):
        
        # This variable stores all the learners in a form of a list
        self.decision_tree_learners = []
        number_of_records = int(len(records)*training_set_percent/100)
        weights = [1/len(records)]*len(records)
        for num in range(number_of_learners):
            # Get the bootstrapped sample from the training dataset
            bagged_records = self.get_bagged_records(records, number_of_records, weights)
            decision_tree = Decision_Tree(bagged_records, None, False)
            self.decision_tree_learners.append(decision_tree)
            weights = decision_tree.perform_weighing(records, weights)
            
    # Function to perform Bootstrap sampling on the training dataset
    def get_bagged_records(self, records, number_of_records, record_weights):
        
        indices_list = random.choices(range(len(records)), weights=record_weights, k=number_of_records)
        [random.choice(range(len(records))) for _ in range(number_of_records)]
        bagged_records = []
        for index in indices_list:
            bagged_records.append(records[index])
        return bagged_records
    
    # Function to traverse all the Decision trees and perform a class prediction using their cummulative value
    def predict_label(self, record):
        
        label_dict = {}
        label_pred = None
        max_weight = None
        for decision_tree in self.decision_tree_learners:
            pred = decision_tree.predict_label(record)
            if pred in label_dict:
                label_dict[pred] = label_dict[pred] + decision_tree.alpha
            else:
                label_dict[pred] = decision_tree.alpha
        for label in label_dict:
            if max_weight == None or max_weight <= label_dict[label]:
                label_pred = label
                max_weight = label_dict[label]
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