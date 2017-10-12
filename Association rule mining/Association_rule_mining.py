"""
Description: Generate Association Rules for a given support count and confidence using Apriori Algorithm.
Group Members: Mitali Bhiwande | Sumedh Ambokar | Tejasvi Sankhe
"""


import numpy as np
import sys


# Function to read the data file. Every gene is represented using proper naming convention. {GNumber_gene}
def read_data(file_name):
    with open(file_name) as input_data:
        genes=[line.replace("\n","").split('\t') for line in input_data]
    dataset_list=[]
    for sample in np.array(genes):
        i = 1
        data_set = set()
        for gene in sample:
            data_set.add("G" + str(i) + "_" + gene)
            i = i + 1
        dataset_list.append(data_set)
    return dataset_list

# Function to get support count(Number of occurences) of a set from the given data set
def get_support_count(input_list, dataset_list):
    count = 0
    for data_set in dataset_list:
        if set(input_list).issubset(data_set):
            count = count + 1
    return count


# Function used to generate new sets of size k + 1 from the previous frequent sets which are of size k.
def generate_sets(size_of_set, previous_freq_dataset_list, dataset_list, support_count):
    i = 0
    list_of_sets = []
    support_count_dic = {}
    flag = False
    if size_of_set == 1:
        dic={}
        for sample in dataset_list:
            for gene in sample:
                if gene in dic:
                    dic.update({gene:dic.get(gene) +1})
                else: 
                    dic.update({gene:1})
        for key in list(dic.keys()):
            if dic.get(key) < support_count:
                del dic[key]  
            else:
                support_count_dic.update({key:dic.get(key)})
        return list(dic), support_count_dic
    else:
        for first_dataset in previous_freq_dataset_list[i:]:
            for second_dataset in previous_freq_dataset_list[i+1:]:
                if size_of_set == 2:
                    pair = []
                    pair.append(first_dataset)
                    pair.append(second_dataset)
                    sc = get_support_count(pair, dataset_list)
                    if sc >= support_count:
                        list_of_sets.append(pair)
                        support_count_dic.update({",".join(pair):sc})
                else:
                    for ele in range(size_of_set - 2):
                        if first_dataset[ele] != second_dataset[ele]:
                            flag = True
                            break
                    if flag:
                        flag = False
                        break
                    else:
                        dataset = []
                        dataset = dataset+first_dataset
                        dataset.append(second_dataset[len(second_dataset) - 1])
                        sc = get_support_count(dataset, dataset_list)
                        if sc >= support_count:
                            list_of_sets.append(dataset)
                            support_count_dic.update({",".join(dataset):sc})
            i = i + 1
    return list_of_sets, support_count_dic


# Function to get all the frequent sets from a dataset for a given support count percentage.
# It also returns a dictionary which contains support count of each frequent set.
def generate_all_frequent_set(support_count_percent, dataset_list):
    support_count_dic = {}
    frequent_data_set = []
    support_count = int(support_count_percent*len(dataset_list)/100)
    freq_set_length_count = 1
    previous_freq_dataset_list = []
    set_size = 1
    while freq_set_length_count > 0:
        previous_freq_dataset_list, support_count_dic_temp = generate_sets(set_size, previous_freq_dataset_list, dataset_list, support_count)
        support_count_dic.update(support_count_dic_temp)
        if set_size > 1:
            frequent_data_set = frequent_data_set + previous_freq_dataset_list
        freq_set_length_count = len(previous_freq_dataset_list)
        set_size += 1
    return frequent_data_set, support_count_dic


# Function to check if a given Association rule has confidence equal to or higher than passed parameter
def validate_rules(body,freq_set,confidence_percent,freq_dataset_dic):
    confidence=freq_dataset_dic.get(freq_set)/(freq_dataset_dic.get(body))
    if confidence*100 >= confidence_percent:
        return True
    else:
        return False


# Function used to generate all possible Association rules for each of the frequent sets passed in a list.
def generate_rules(input_set_list,confidence_percent,freq_dataset_dic):
    head_return=[]
    body_return=[]
    for input_list in input_set_list:
        head=[]
        body=[]
        for i in range(len(input_list)):
            input_copy = input_list[:]
            h=input_list[i:i+1]
            h=",".join(h)
            del input_copy[i]
            b=input_copy
            b=",".join(b)
            freq_set_str = ",".join(input_list)
            if validate_rules(b,freq_set_str,confidence_percent,freq_dataset_dic):
                if h not in head:
                    head.append(h)
                    body.append(b)
            if validate_rules(h,freq_set_str,confidence_percent,freq_dataset_dic):
                if h not in body: 
                    head.append(b)
                    body.append(h)
            if len(input_list)>2:
                for j in range(i+1, len(input_list)):
                    temp_list=[]   
                    input_copy = input_list[:]
                    temp_list.append(input_list[i])
                    temp_list.append(input_list[j])
                    input_copy.remove(input_list[i])
                    input_copy.remove(input_list[j])
                    temp_list=",".join(temp_list)
                    input_copy=",".join(input_copy)
                    if validate_rules(input_copy,freq_set_str,confidence_percent,freq_dataset_dic):
                        if temp_list not in head:
                            head.append(temp_list)
                            body.append(input_copy)
                    if validate_rules(temp_list,freq_set_str,confidence_percent,freq_dataset_dic):
                        if temp_list not in body:
                            body.append(temp_list)
                            head.append(input_copy)
                 
        head_return = head_return + head
        body_return = body_return + body
    return head_return, body_return


# Function to handle template1 for generation of association rules.
def template1(rule,number,genes,rules_head_String,rules_body_String):
    res=[]
    count=0
    if number=="ANY":
        for s in genes:
            if rule=="RULE":
                for i in range(0,len(rules_head_String)):
                    if s in rules_head_String[i]:
                        temp = rules_body_String[i]+' ==> '+rules_head_String[i]
                        res.append(temp)
                        count+=1
                    elif s in rules_body_String[i]:
                        temp = rules_body_String[i]+' ==> '+rules_head_String[i]
                        res.append(temp)
                        count+=1  
            elif rule=="BODY":
                for i in range(0,len(rules_body_String)):
                    if s in rules_body_String[i]:
                        temp = rules_body_String[i]+' ==> '+rules_head_String[i]
                        res.append(temp)
                        count+=1 
            elif rule=="HEAD":
                for i in range(0,len(rules_head_String)):
                    if s in rules_head_String[i]:
                        temp = rules_body_String[i]+' ==> '+rules_head_String[i]
                        res.append(temp)
                        count+=1
    elif number=='NONE':
        index_to_remove=[]
        for s in genes:
            if rule=="RULE":
                for i in range(0,len(rules_head_String)):
                    if s in rules_head_String[i]:
                        index_to_remove.append(i)
                    elif s in rules_body_String[i]:
                        index_to_remove.append(i)   
            elif rule=="BODY":
                for i in range(0,len(rules_body_String)):
                    if s in rules_body_String[i]:
                        index_to_remove.append(i) 
            elif rule=="HEAD":
                for i in range(0,len(rules_head_String)):
                    if s in rules_head_String[i]:
                        index_to_remove.append(i)
        for i in range(0,len(rules_body_String)):
            if i not in index_to_remove:
                        temp = rules_body_String[i]+' ==> '+rules_head_String[i]
                res.append(temp)
                count+=1
    elif number=="1":
        result=[];
        for s in genes:
            if rule=="RULE":
                for i in range(0,len(rules_head_String)):
                    if s in rules_head_String[i]:
                        temp = rules_body_String[i]+' ==> '+rules_head_String[i]
                        result.append(temp)
                    elif s in rules_body_String[i]:
                        temp = rules_body_String[i]+' ==> '+rules_head_String[i]
                        result.append(temp) 
            elif rule=="BODY":
                for i in range(0,len(rules_head_String)):
                    if s in rules_body_String[i]:
                        temp = rules_body_String[i]+' ==> '+rules_head_String[i]
                        result.append(temp) 
            elif rule=="HEAD":
                for i in range(0,len(rules_head_String)):
                    if s in rules_head_String[i]:
                        temp = rules_body_String[i]+' ==> '+rules_head_String[i]
                        result.append(temp)
        if rule=='RULE':
            for lines in result:
                gene_counter=0;
                for s in genes:
                    if s in lines:  
                        gene_counter+=1;        
                if gene_counter<=1:
                    res.append(lines)
                    count+=1
        elif rule=='HEAD':
            for lines in result:
                    gene_counter=0;
                    for s in genes:
                        if s in lines.split(' ==> ')[0]:
                            gene_counter+=1;
                    if gene_counter<=1:
                        res.append(lines)
                        count+=1
        elif rule=='BODY':
            for lines in result:
                    gene_counter=0;
                    for s in genes:
                        if s in lines.split(' ==> ')[1]:
                            gene_counter+=1;
                    if gene_counter<=1:
                        res.append(lines)
                        count+=1      
    return res,count


# Function to handle template2 for generation of association rules.
def template2(rule, number, placeHolder,rules_head_String,rules_body_String):
    #call to generate rules
    #count G's in string and compare count with number passed
    res=[]
    count=0
    if rule=="RULE":
        for i in range(0,len(rules_body_String)):
            if rules_head_String[i].count('G')+rules_body_String[i].count('G')>=number:
                res.append(rules_body_String[i]+' ==> '+rules_head_String[i])
                count+=1
    elif rule=='HEAD':
        for i in range(0,len(rules_head_String)):
            if rules_head_String[i].count('G')>=number:
                res.append(rules_body_String[i]+' ==> '+rules_head_String[i])
                count+=1
    elif rule=='BODY':
        for i in range(0,len(rules_body_String)):
            if rules_body_String[i].count('G')>=number:
                res.append(rules_body_String[i]+' ==> '+rules_head_String[i])
                count+=1
    return res,count


# Function to handle template3 for generation of association rules.
def template3(condition,rule1,number1,gene1,rule2,number2,gene2,rules_head_String,rules_body_String):
    ansSet = set()
    if condition=='1or1':
        temp1=template1(rule1,number1,gene1,rules_head_String,rules_body_String)
        temp2=template1(rule2,number2,gene2,rules_head_String,rules_body_String)
        ansSet=set(temp1[0]).union(set(temp2[0]))
    elif condition=='1and1':
        temp1=template1(rule1,number1,gene1,rules_head_String,rules_body_String)
        temp2=template1(rule2,number2,gene2,rules_head_String,rules_body_String)
        ansSet=set(temp1[0]).intersection(set(temp2[0]))
    elif condition=='1or2':
        temp1=template1(rule1,number1,gene1,rules_head_String,rules_body_String)
        temp2=template2(rule2,number2,gene2,rules_head_String,rules_body_String)
        ansSet=set(temp1[0]).union(set(temp2[0]))
    elif condition=='1and2':
        temp1=template1(rule1,number1,gene1,rules_head_String,rules_body_String)
        temp2=template2(rule2,number2,gene2,rules_head_String,rules_body_String)
        ansSet=set(temp1[0]).intersection(set(temp2[0]))
    elif condition=='2or2':
        temp1=template2(rule1,number1,gene1,rules_head_String,rules_body_String)
        temp2=template2(rule2,number2,gene2,rules_head_String,rules_body_String)
        ansSet=set(temp1[0]).union(set(temp2[0]))
    elif condition=='2and2':
        temp1=template2(rule1,number1,gene1,rules_head_String,rules_body_String)
        temp2=template2(rule2,number2,gene2,rules_head_String,rules_body_String)
        ansSet=set(temp1[0]).intersection(set(temp2[0]))
    return ansSet,len(ansSet)


# Function to print the generated Association Rules
def print_assoc_rule(results):
    for line in results[0]:
        print(line)
    print("Count --> ", results[1])


# Reads parameters from the command line inputs and call the appropriate function.	
if len(sys.argv) > 1:	
	dataset_list = read_data(sys.argv[1])
	support = int(sys.argv[2])
	confidence = int(sys.argv[3])
	frequent_data_set, support_count_dic = generate_all_frequent_set(support, dataset_list)
	rules_head_String, rules_body_String = generate_rules(frequent_data_set, confidence, support_count_dic)
	if sys.argv[4] == 'template1':
		rules = template1(sys.argv[5], sys.argv[6], sys.argv[7].split(","), rules_head_String, rules_body_String)
		print_assoc_rule(rules)
	if sys.argv[4] == 'template2':
		rules = template2(sys.argv[5], int(sys.argv[6]), "", rules_head_String, rules_body_String)
		print_assoc_rule(rules)
	if sys.argv[4] == 'template3':
		if sys.argv[5] == '1or1' or sys.argv[5] == '1and1':
			rules = template3(sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8].split(","), sys.argv[9], sys.argv[10], sys.argv[11].split(","), rules_head_String, rules_body_String)
			print_assoc_rule(rules)
		elif sys.argv[5] == '1or2' or sys.argv[5] == '1and2':
			rules = template3(sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8].split(","), sys.argv[9], int(sys.argv[10]), "", rules_head_String, rules_body_String)
			print_assoc_rule(rules)
		elif sys.argv[5] == '2or2' or sys.argv[5] == '2and2':
			rules = template3(sys.argv[5], sys.argv[6], int(sys.argv[7]), "", sys.argv[8], int(sys.argv[9]), "", rules_head_String, rules_body_String)
			print_assoc_rule(rules)
else:
	print("Not sufficient Inputs")