import first_file as ff
import second_file as sf
import third_file as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_and_pred():
    # Function to load the data set and split randomly for perfect functioning
    cancer=datasets.load_wine()
    #C ode executed for wine database
    x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=1)
    # print(cancer)
    # print(x_train)
    # print(len(y_train),y_train.shape)
    decision_tree_fit(x_train,y_train)

# This function is used to split the incoming parent node into two child nodes (continuous data) on the basis of gain ratio.
def split_node(x,y,feature):
    temp=0
    flag6=False
    val = np.zeros(x.shape[1])
    gain = np.ones(x.shape[1])
    gain=gain*-1
    for i in range(x.shape[1]):                                 # Value Calculation
        flag6=False                                             # The following conditions check when to end the splitting of data.
        flag1 = tf.pure_node_check(y)
        # flag2 = tf.feature_split_check(feature, x.shape[1])
        # print(x)
        flag2=False
        if flag1 == True and flag2 == False:
            # print("Reached Pure/Leaf Node")
            temp=1
        elif flag1 == False and flag2 == True:
            # print("All Features on which Split could happen is complete")
            temp = 1
        elif flag1 == True and flag2 == True:
            # print("Reached Pure/Leaf Node")
            temp = 1
        else:
            for k in feature:                               # To ensure a single feature is used for a single level.
                if i==k:
                    flag6=True
                    break
            if flag6==True:
                continue
            else:
                val[i] = sf.value_calculation(x[:, i], y)           # Calculation of optimum value for a continuous range of a data
                split_1, split_2 = sf.split(x[:, i], val[i])
                gain[i] = tf.gain_ratio(split_1, split_2, y)        # Calculation of gain_ratio for the splits.
                # print(val)
            max_gain = max(gain)
            for i in range(len(gain)):
                if max_gain == gain[i]:
                    feature.append(i)
                    break
            # print(val[feature_val])
            new_node_x1, new_node_x2 = sf.split(x[:, feature[len(feature)-1]], val[feature[len(feature)-1]])
            new_node_1 = []
            y_new_1 = []
            new_node_2 = []
            y_new_2 = []
            for i in range(len(new_node_x1)):
                new_node_1.append(x[new_node_x1[i]])
                y_new_1.append(y[new_node_x1[i]])
            new_node_1 = np.array(new_node_1)
            y_new_1 = np.array(y_new_1)
            for i in range(len(new_node_x2)):
                new_node_2.append(x[new_node_x2[i]])
                y_new_2.append(y[new_node_x2[i]])
            new_node_2 = np.array(new_node_2)
            y_new_2 = np.array(y_new_2)
            # print(len(new_node_2),len(new_node_1))
        if temp==1:
            # print(temp)
            return 0
        else:
            return new_node_1, new_node_2, y_new_1, y_new_2, feature

# This function creates a tree-like data structure and stores nodes after splitting them.
def decision_tree_fit(x, y):
    level = 0
    c=0
    flag5=False
    temp=[]
    feature=[]
    tree_x = []                     # Holds the x part of the tree
    tree_y = []                     # Holds the y part of the tree
    for i in range(2**x.shape[1]):
        tree_x.append(0)
        tree_y.append(0)
    tree_x[0]=(x)
    tree_y[0]=(y)
    for i in range(len(tree_x)):        # Here to run through the tree we display each node level by level rather than inorder/postorder traversals.
        # print((tree_x[i]))
        # print("c=",c)
        if i == (2 ** level):
            level = level + 1
        elif i==0:
            level = level + 1
        if i in temp:
            # This ensures that after reaching a pure node the code doesn't try to split the pure node by storing the non important position indexes in temp array.
            c=c+1
            temp.append(2 * i + 1)
            temp.append(2 * i + 2)
            continue
        flag1 = tf.pure_node_check(tree_y[i])
        flag2 = tf.feature_split_check(feature, x.shape[1])
        if flag1 == True and flag2 == False:
            ff.node_information(tree_x[i], tree_y[i], level,feature)
            c=c+1
            temp.append(2 * i + 1)
            temp.append(2 * i + 2)
        elif flag1 == False and flag2 == True:
            ff.node_information(tree_x[i], tree_y[i], level,feature)
            c=c+1
            temp.append(2 * i + 1)
            temp.append(2 * i + 2)
        elif flag1 == True and flag2 == True:
            ff.node_information(tree_x[i], tree_y[i], level,feature)
            c=c+1
            temp.append(2 * i + 1)
            temp.append(2 * i + 2)
        else:
            ff.node_information(tree_x[i], tree_y[i], level,feature)
            node1,node2,y1,y2,feature=split_node(tree_x[i], y,feature)
            tree_x[2 * i + 1]=(node1)                                   # Storing left node on 2n+1 position and right node in the 2n+2 position.
            tree_y[2 * i + 1]=(y1)
            tree_x[2 * i + 2]=(node2)
            tree_y[2 * i + 2]=(y2)
            c=c+1
        if level==x.shape[1]:                       # The code is run (no.of features) times. Due to gain ratio the splits are resulting into highly pure and impure nodes.
            break                                   # So it needs to be run multiple times.

load_and_pred()
