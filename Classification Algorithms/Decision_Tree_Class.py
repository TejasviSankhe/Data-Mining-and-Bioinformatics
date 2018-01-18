############### Decision Tree Class #############

import random
import math

class Decision_Tree:
    
    # Class to store the condition for a decision node
    class Decision:
    
        def __init__(self, attribute_number, value):
            self.attribute_number = attribute_number
            try:
                self.value = float(value)
            except ValueError:
                self.value = value

        def compare(self, input):
            try:
                val = float(input)
                return val >= self.value
            except ValueError:
                return input == self.value

        def __repr__(self):
            try:
                float(self.value)
                return "is column number " + str(self.attribute_number) + " >= " + str(self.value) + "?"
            except ValueError:
                return "is column number " + str(self.attribute_number) + " == " + str(self.value) + "?"

    # Class used to store the independent Nodes in the Decision Tree
    class Node:
    
        def __init__(self, decision, true_node, false_node, node_type, node_label):
            self.decision = decision
            self.false_node = false_node
            self.true_node = true_node
            self.type = node_type
            self.label = node_label
    
    # Function to calculate Gini Index for a given list of records
    def calculate_gini_index(self, records):
    
        dict = {}
        # get count against ever label
        for record in records:
            if record[-1] in dict:
                dict[record[-1]] = dict[record[-1]] + 1
            else:
                dict[record[-1]] = 1

        # calculate gini index by adding sum of squared probability of every label and subtracting it from 1
        return 1 - sum((count/len(records))**2 for key, count in dict.items())
    
    # Function to find the best feature value to split the data on
    def find_best_decision(self, records, attributes_set, is_random_forest):
    
        parent_gini_index = self.calculate_gini_index(records)
        max_gain = 0
        final_decision = None
        final_true_records = None
        final_false_records = None
        attributes = None
        if is_random_forest:
            attributes = attributes_set[random.choice(range(len(attributes_set)))]
        else:
            attributes = range(len(records[0]) - 1)
        for i in attributes:
            unique_values = set([record[i] for record in records])
            for value in unique_values:
                decision = self.Decision(i, value)
                true_count = 0
                gini_index = 0
                gain = 0
                true_records = []
                false_records = []
                for record in records:
                    if decision.compare(record[i]):
                        true_records.append(record)
                    else:
                        false_records.append(record)
                gain = parent_gini_index - (len(true_records)/len(records)*self.calculate_gini_index(true_records) + len(false_records)/len(records)*self.calculate_gini_index(false_records))
                if gain > max_gain:
                    max_gain = gain
                    final_decision = decision
                    final_true_records = true_records
                    final_false_records = false_records
        return max_gain, final_decision, final_true_records, final_false_records
    
    # Function to recursively construct a decision tree
    def build_decision_tree(self, records, attributes_set, is_random_forest):
    
        gain, decision, true_records, false_records = self.find_best_decision(records, attributes_set, is_random_forest)

        if(gain == 0):
            n = self.Node(None, None, None, 'Label_Node', records[0][-1])
            return n

        true_branch = self.build_decision_tree(true_records, attributes_set, is_random_forest)
        false_branch = self.build_decision_tree(false_records, attributes_set, is_random_forest)

        return self.Node(decision, true_branch, false_branch, 'Decision_Node', None)
    
	# Function used to recursively iterate on the Decision_Tree to identify the prediction of a test record
    def make_prediction(self, record, node):
    
        if node.type == "Label_Node":
            return node.label
        if node.decision.compare(record[node.decision.attribute_number]):
            return self.make_prediction(record, node.true_node)
        return self.make_prediction(record, node.false_node)
    
    # Function to print the Decision_Tree using recursive approach
    def print_tree(self, node, spacing=""):
        """World's most elegant tree printing function."""

        # Base case: we've reached a Label Node/Leaf
        if node.type == 'Label_Node':
            print (spacing + " Pediction --> ", node.label)
            return

        # Print the question at this node
        print (spacing + str(node.decision))

        # Call this function recursively on the true branch
        print (spacing + '--> True Branch:')
        self.print_tree(node.true_node, spacing + "  	")

        # Call this function recursively on the false branch
        print (spacing + '--> False Branch:')
        self.print_tree(node.false_node, spacing + "  	")
    
    def __init__(self, records, attributes, is_random_forest):
        
        self.head_decision_node = self.build_decision_tree(records, attributes, is_random_forest)
        self.error = None
        self.alpha = None
    
    # Function to calculate performance measures of Decision Tree
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
    
    def predict_label(self, record):
        return self.make_prediction(record, self.head_decision_node)
    
    def print_decision_tree(self):
        
        self.print_tree(self.head_decision_node, spacing="")
        
    # Function used to update the weights corresponding the training records after classifying them through the Decision Tree learner
    def perform_weighing(self, records, weights):
        
        error = 0
        pred_list = []
        for i in range(len(records)):
            pred = self.make_prediction(records[i][0:-1], self.head_decision_node)
            pred_list.append(pred)
            if records[i][-1] != pred:
                error = error + weights[i]
        self.error = error/sum(weights)
        self.alpha = 0.5*math.log((1-self.error)/self.error)
        for i in range(len(weights)):
            if pred_list[i] == records[i][-1]:
                weights[i] = weights[i]*math.exp(-self.alpha)
            else:
                weights[i] = weights[i]*math.exp(self.alpha)
        return weights