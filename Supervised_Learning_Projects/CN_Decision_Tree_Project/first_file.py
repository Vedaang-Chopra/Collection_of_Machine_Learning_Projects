import second_file as sf
import third_file as tf
import math
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def count_types_output(y):              # This function calculates the different types of
    a=[]                                # outputs that are present in the output column.
    for i in y:                         # eg:- It will calculate and result [0,1](malignant/bennie) for cancer
        if i in a:                      # data set and [0,1](survived/not survived) for the titanic data set.
            continue
        else:
            a.append(i)
    return a


def output_counter(y):                              # This function calculates the number of occurrences of the
    output_label = count_types_output(y)            # different types of outputs present in the output column.
    output_label=np.array(output_label)
    # print(output_label.shape)
    label_count = np.zeros(output_label.shape)
    for i in range(0,len(y)):
        for j in range(len(output_label)):
            if y[i] == output_label[j]:
                label_count[j] = label_count[j] + 1
    # print(label_count)
    return label_count


def entropy(y):                                     # Function to calculate the entropy
    count=output_counter(y)
    # print(count)
    prob=np.array(count)
    prob=prob/prob.sum()
    # print(prob)
    for i in range(0,len(prob)):
        a=math.log(prob[i],10)
        prob[i]=prob[i]*a
    prob = prob * (-1)
    # print(prob)
    entropy_val=prob.sum()
    return entropy_val


def node_information(x,y,level,feature):
    # For any single node of the entire tree structure we have a few parameters associated with itsuch as
    # 1.The Total elements
    # 2.The different types of outputs and their no. of occurrences
    # 3.The Entropy
    # 4.The Gain Ratio
    # 5.The Level of the node
    # This function is used to print that node information.
    total_elements = len(x)
    y_new=y
    y_new=np.array(y_new)
    # print(y_new)
    count_output=output_counter(y_new)
    entropy_val=entropy(y_new)
    print("Level is :",level)
    a=(count_types_output(y_new))
    a=np.array(a)
    for i in range(len(a)):
        print("Count of",a[i]," is :",count_output[i])
    print("Entropy is is :", entropy_val)
    # print("PPure Node Check",count_output)
    flag1=tf.pure_node_check(y)
    # flag2=tf.feature_split_check(feature,x.shape[1])
    # print(x)
    flag2=False
    if flag1==True and flag2==False:
        print("Reached Pure/Leaf Node")
    elif flag1==False and flag2==True:
        print("All Features on which Split could happen is complete")
    elif flag1==True and flag2==True:
        print("Reached Pure/Leaf Node")
    else:
        val = np.zeros(x.shape[1])
        gain = np.zeros(x.shape[1])
        for i in range(x.shape[1]):                                     # Value Calculation
            if i in feature:
                continue
            else:
                val[i] = sf.value_calculation(x[:, i], y)
                split_1, split_2 = sf.split(x[:, i], val[i])
                gain[i] = tf.gain_ratio(split_1, split_2,y)
        # print(gain)
        max_gain = max(gain)
        feature_val = -1
        for i in range(len(gain)):
            if max_gain == gain[i]:
                feature_val = i
                break
        print("Splitting on feature", feature_val, "with gain ratio", max_gain)

