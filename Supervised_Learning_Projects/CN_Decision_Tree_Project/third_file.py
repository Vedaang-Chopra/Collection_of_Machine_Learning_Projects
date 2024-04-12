# This file deals with calculation of information gain and gain ratio of the data provided to it.
import first_file as ff
import math
import numpy as np
import mpmath as mp

def parent_node_information_gain(y):
    parent_count=ff.output_counter(y)
    parent_entropy=ff.entropy(parent_count)
    # print(parent_entropy)
    parent_information_gain = ((parent_count.sum() / len(y)) * parent_entropy)
    return parent_information_gain

def information_gain(split_1,split_2,y):
    y_new_1 = []
    y_new_2 = []
    for i in range(len(split_1)):
        y_new_1.append(y[split_1[i]])
    y_new_1 = np.array(y_new_1)
    for i in range(len(split_2)):
        y_new_2.append(y[split_2[i]])
    y_new_2 = np.array(y_new_2)
    count_1 = ff.output_counter(y_new_1)
    count_2 = ff.output_counter(y_new_2)
    # print(count_1.sum()+count_2.sum(),"c")
    entropy_val_1 = ff.entropy(count_1)
    entropy_val_2 = ff.entropy(count_2)
    count_1 = np.array(count_1)
    count_2 = np.array(count_2)
    length=count_2.sum()+count_1.sum()
    temp1=((count_1.sum()/length)* entropy_val_1)
    information_req_val=temp1+((count_2.sum()/length) * entropy_val_2)
    information_gain_val=parent_node_information_gain(y)-information_req_val
    return information_gain_val

def gain_ratio(split_1,split_2,y):
    information_gain_val=information_gain(split_1,split_2,y)
    # print(information_gain_val)
    y_new_1 = []
    y_new_2 = []
    for i in range(len(split_1)):
        y_new_1.append(y[split_1[i]])
    y_new_1 = np.array(y_new_1)
    for i in range(len(split_2)):
        y_new_2.append(y[split_2[i]])
    y_new_2 = np.array(y_new_2)
    count_1 = ff.output_counter(y_new_1)
    count_2 = ff.output_counter(y_new_2)
    count_1 = np.array(count_1)
    count_2 = np.array(count_2)
    temp1,temp2=0,0
    if count_1.size == 0:
        temp2 = ((count_2.sum() / len(y)))
        temp2 = temp2 * (math.log(temp2, 10))
        temp2 = temp2 * -1
        temp1=0
    elif count_2.size == 0:
        temp1 = ((count_1.sum() / len(y)))
    # print(temp1)
        temp1= temp1 * (math.log(temp1,10))
        temp1=temp1*-1
    else:
        temp1 = ((count_1.sum() / len(y)))
        # print(temp1)
        temp1 = temp1 * (math.log(temp1, 10))
        temp1 = temp1 * -1
        temp2 = ((count_2.sum() / len(y)))
        temp2 = temp2 * (math.log(temp2, 10))
        temp2 = temp2 * -1
    split_info = temp1+temp2
    return  mp.mpf(information_gain_val/split_info)


def pure_node_check(y):
    # This function is used to check that whether the further splitting of the node is possible or not by checking if the class is pure or not.
    count = ff.output_counter(y)
    # print(count)
    flag = True
    c = 0
    for i in range(len(count)):
        if count[i] == 0:
            flag = True
        elif count[i] != 0 and c <= 1:
            flag = True
            c = c + 1
        elif count[i] != 0 and c > 1:
            flag = False
            break
        # print(flag)
    if c>1:
        flag=False
    return flag
# This function checks whether the number of features on which data is split is equal to the number of features.
def feature_split_check(feature,x_shape):
    if len(feature)== x_shape:
        flag=True
    else:
        flag=False
    return flag