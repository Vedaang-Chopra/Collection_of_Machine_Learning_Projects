import first_file as ff
import numpy as np

# Split Function is used to split the data into two.
def split(x, val):
    a1 = []
    a2 = []
    for i in range(len(x)):
        if x[i] <= val:
            a1.append(i)
        else:
            a2.append(i)
    return a1, a2

#
def accuracy_column(x, y, mid_element,x_copy):
    # For a single column with continious range data, we calculate a value which results in the max accuracy.
    # This function is used to calculate that value.
    split_1,split_2=split(x_copy,mid_element)
    # print(x_copy)
    # print(len(split_1),len(split_2))
    y_new_1 = []
    y_new_2 = []
    check=ff.count_types_output(y)
    # print(check)
    # print(len(split_1),len(split_2))
    for i in range(len(split_1)):
        # print(split_1[i],y[split_1[i]])
        y_new_1.append(y[split_1[i]])
    y_new_1 = np.array(y_new_1)
    for i in range(len(split_2)):
        y_new_2.append(y[split_2[i]])
    y_new_2 = np.array(y_new_2)
    # print(len(y_new_1) + len(y_new_2) == 133)
    # print(len(y_new_1))
    # print(len(y_new_2))
    count_1 = ff.output_counter(y_new_1)
    count_2 = ff.output_counter(y_new_2)
    # print((count_1),count_2)

    total = count_1.sum() + count_2.sum()
    if count_1.size==0:
        accuracy_val =max(count_2)/total
    elif count_2.size==0:
        accuracy_val = max(count_1) / total
    else:
        accuracy_val=(max(count_1)+max(count_2))/total
    # print(accuracy_val)
    return accuracy_val,mid_element

# The Value_Calculation function is used to calculate the value

def value_calculation(x, y):
    accuracy_val= []
    mid_element_arr=[]
    x_copy=x.copy()
    # print(x)
    x.sort()
    # print(x_copy)
    # print(len(x))
    # print(x)
    for i in range(len(x)):
        if i == len(x) - 1:
            break
        else:
            mid_element = (x[i] + x[i + 1]) / 2
            # print(x[i],x[i+1],mid_element)
            # print(accuracy_column(x, y, mid_element))
            a1,a2=(accuracy_column(x, y, mid_element,x_copy))
            accuracy_val.append(a1)
            mid_element_arr.append(a2)
        accuracy_value=max(accuracy_val)
        for i in range(len(accuracy_val)):
            if accuracy_val[i]==accuracy_value:
                return mid_element_arr[i]

