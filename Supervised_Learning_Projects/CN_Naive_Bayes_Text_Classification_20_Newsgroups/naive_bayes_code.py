import numpy as np
import math
import os

# The following function finds the p(i)/probability of each group/class.
def basic_probability_calc(class_label):
    basic_prob=np.zeros(len(class_label))
    for i in range(len(class_label)):
        path_main = 'G:\\Development\\Projects\\Python Projects\\Machine_Learning\\Naive_Bayes\\Naive_Bayes_Project\\Train_Data\\'
        path_main = path_main + class_label[i]
        os.chdir(path_main)
        path_main = path_main + '\\'
        files = os.listdir()
        # print(len(files))
        basic_prob[i]=len(files)
    basic_prob=basic_prob/basic_prob.sum()
    return basic_prob

# The following function implements the multinomial bayes theorem for any file.
# It divides the count of that word(found in x_test file) by the corresponding sum of all the words for that group/class.
# It then adds the log of each division and gives the result by adding with the log of the probabiltiy of that class.
# It then returns the max probabilty class.
def naive_bayes(x_train,y_train,x_test):
    # print(len(x_train))
    prob = basic_probability_calc(y_train)
    prob=np.array(prob)
    for i in prob:
        i=math.log(i,2)
    c=np.zeros(len(x_train)-1)
    sum=np.zeros(len(x_train)-1)
    temp=0
    # The corresponding loop finds the sum of all words corresponding to that label.
    for i in range(1,len(x_train)):
        for j in range(len(x_train[i])):
            sum[i-1]=sum[i-1]+int(x_train[i][j])
    # print(sum)
    # print(x_test)
    # print(x_train)
    for i in range(1,len(x_train)):
        # print("Next Output Label")
        for j in range(len(x_test)):
            if x_test[j] > 0:
                # print(temp,x_test[j],x_train[i][j],sum[i-1])
                temp_value=((int(x_train[i][j])+1)/(sum[i-1]+(x_train.shape[1])))
                temp=temp+math.log(temp_value,2)
        c[i-1]=temp
        temp=0
    c=prob+c
    max_value=max(c)
    # print(c)
    for i in range(len(c)):
        if c[i]==max_value:
            return y_train[i]

