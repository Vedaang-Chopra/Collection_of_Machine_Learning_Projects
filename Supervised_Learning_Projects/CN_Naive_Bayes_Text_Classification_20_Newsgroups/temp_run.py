import sys
import shutil
import math
import numpy as np
from nltk.corpus import stopwords
import os
import random


# This file creates the feature set for the documents given.
def creating_dataset(vocab,output_labels,word_array,freq_array):
    # main_content=[]
    content=[[0 for j in range(len(vocab))] for i in range(len(output_labels))]
    for i in range(len(word_array)):
        for j in range(len((word_array[i]))):
            for k in range(len(vocab)):
                if word_array[i][j]==vocab[k]:
                    content[i][k]=freq_array[i][j]
                else:
                    continue
    return content

# The following creates the vocabulary for all the words.
def forming_vocab(word_array,freq_array):
    val=4
    freq_array=np.array(freq_array)
    word_array = np.array(word_array)
    # freq_array_copy=freq_array.copy()
    # for i in range(len(freq_array)):
    #     freq_array_copy[i].sort()
    # print(freq_array_copy)
    temp_holder=[]
    # print(freq_array.shape[0])
    for i in range(len(freq_array)):
        for j in range(len(freq_array[i])):
            # print(freq_array[i][j])
            if freq_array[i][j]<=val:
                temp_holder.append((i,j))
            else:
                continue
    vocab=[]
    for i in range((word_array).shape[0]):
        for j in range(len(word_array[i])):
            temp=(i,j)
            if temp in temp_holder:
                continue
            else:
                vocab.append(word_array[i][j])
    return vocab

# The function reads all the files word by word, finds the frequency of each word, and appends the word into an array
# so that the top k words can be found for creating vocabulary.
def reading_files(class_label):
    pathname = os.path.dirname(sys.argv[0])
    path_temp=os.path.abspath(pathname)
    path_main = path_temp+'\\Train_Data\\'
    path_main = path_main + class_label
    os.chdir(path_main)
    stop_words_list = stopwords.words("english")
    path_main = path_main + '\\'
    files = os.listdir()
    word_array = []
    freq_array = []
    for k in range(len(files)):         # Reduce this value if code takes a lot of time
        fr = open(path_main + str(files[k]), 'r')
        string = fr.read()
        string.strip()
        word = ""
        for i in range(len(string)):
            if string[i] == " " or string[i] == "," or string[i] == ":" or string[i] == "\"" or string[i] == "(" or string[i] == ")" or string[i] == "|" or string[i] == "-" or string[i] == "." or string[i] == "{" or string[i] == "}" or string[i] == "[" or string[i] == "]" or string[i] == "@" or string[i] == "\n" or string[i] == ">" or string[i] == "<" or string[i] == "!":
                # print(word)
                word=(((word).strip()).lower())
                if word in stop_words_list:
                    word=""
                elif word==" " or word=="" or word=="," or word==":" or word=="\"" or word=="(" or word==")" or word=="|" or word=="." or word==";" or word=="[" or word=="]" or word==">" or word=="<" or word=="\n" or word=="!" :
                    word=""
                elif word == "{" or word == "}" or word == ">=" or word == "=>" or word == "!=" or word == "<=" or word == "=<" or word == "*" or word == "+" or word == "/" or word == "-" or word == "?" or word == "/" or word == "\\" or word == "@" or word=="=" or word==">" or word=="<" or word=="\'" :
                    word = ""
                elif len(word)==1:
                    if ord(word)>=65 and ord(word)<=90:
                        word=""
                    elif ord(word)>=97 and ord(word)<=122:
                        word=""
                    elif ord(word)>=48 and ord(word)<=57:
                        word=""
                    elif word== "\n":
                        word=""
                    else:
                        word=""
                elif word in word_array:
                    for j in range(len(word_array)):
                        if word_array[j] == word:
                            freq_array[j] = freq_array[j] + 1
                            word = ""
                            break
                else:
                    word_array.append(word)
                    freq_array.append(1)
                    word = ""
            else:
                word = word + string[i]

        # print(len(word_array))
        # print(freq_array)
        fr.close()
    return word_array, freq_array


def fetching_data(output_labels):
    word_array=[[] for i in range(len(output_labels))]
    freq_array = [[] for i in range(len(output_labels))]
    for i in range(len(output_labels)):
        word_array[i],freq_array[i]=(reading_files(output_labels[i]))
    # check_vocab(word_array,freq_array)
    return word_array,freq_array

# Splitting the training and testing Data..............
def train_test_split(y_train):
    # Creates 500 test files ..........................
    for i in range(500):
        random_no = random.randint(0, 19)
        pathname = os.path.dirname(sys.argv[0])
        path_temp = os.path.abspath(pathname)
        path_main = path_temp + '\\Train_data'
        moving = path_temp+'\\test_data'
        path_main = path_main + '\\'
        moving=moving+'\\'
        path_main = path_main + y_train[random_no]
        moving = moving + y_train[random_no]
        os.chdir(path_main)
        path_main = path_main + '\\'
        moving = moving + '\\'
        files = os.listdir()
        no = random.randint(0, len(files))  # Randomizing file selection
        x_test_file_path = path_main + str(files[no])
        moving_path=moving+str(files[no])
        shutil.move(x_test_file_path, moving_path)

def load():
    # print("First File working")
    output_labels = ['alt.atheism',
                     'comp.graphics',
                     'comp.os.ms-windows.misc',
                     'comp.sys.ibm.pc.hardware',
                     'comp.sys.mac.hardware',
                     'comp.windows.x',
                     'misc.forsale',
                     'rec.autos',
                     'rec.motorcycles',
                     'rec.sport.baseball',
                     'rec.sport.hockey',
                     'sci.crypt',
                     'sci.electronics',
                     'sci.med',
                     'sci.space',
                     'soc.religion.christian',
                     'talk.politics.guns',
                     'talk.politics.mideast',
                     'talk.politics.misc',
                     'talk.religion.misc']

    # print(ord("x"))
    train_test_split(output_labels)
    word_array,freq_array=fetching_data(output_labels)
    # stop_words_list = stopwords.words("english")
    vocab=forming_vocab(word_array,freq_array)
    # print(len(vocab))
    content=creating_dataset(vocab,output_labels,word_array,freq_array)
    x=[]
    x.append(vocab)
    for i in range(len(content)):
        x.append(content[i])
    # x=pd.DataFrame
    # x.columns=[vocab[i] for i in range(len(vocab))]
    x=np.array(x)
    # print(x.shape)
    return x,output_labels



# The following function finds the p(i)/probability of each group/class.
def basic_probability_calc(class_label):
    basic_prob=np.zeros(len(class_label))
    for i in range(len(class_label)):
        pathname = os.path.dirname(sys.argv[0])
        path_temp = os.path.abspath(pathname)
        path_main = path_temp+'\\Train_Data\\'
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

# The following Function is used to clean the x_test file by finding the words and their frequency from the files
# group by group(Group here represents a single type of output).
def creating_words(file_path):
    stop_words_list = stopwords.words("english")
    word_array = []
    freq_array = []
    fr = open(file_path, 'r')   # Opening x_test file
    string = fr.read()
    string.strip()
    word = ""
    # The following for loop reads file word by word, removing the stop words and punctuations.
    for i in range(len(string)):
        if string[i] == " " or string[i] == "," or string[i] == ":" or string[i] == "\"" or string[i] == "(" or string[
            i] == ")" or string[i] == "|" or string[i] == "-" or string[i] == "." or string[i] == "{" or string[
            i] == "}" or string[i] == "[" or string[i] == "]" or string[i] == "@" or string[i] == "\n" or string[
            i] == ">" or string[i] == "<" or string[i] == "!":
            # print(word)
            word = (((word).strip()).lower())
            if word in stop_words_list:
                word = ""
            elif word == " " or word == "" or word == "," or word == ":" or word == "\"" or word == "(" or word == ")" or word == "|" or word == "." or word == ";" or word == "[" or word == "]" or word == ">" or word == "<" or word == "\n" or word == "!":
                word = ""
            elif word == "{" or word == "}" or word == ">=" or word == "=>" or word == "!=" or word == "<=" or word == "=<" or word == "*" or word == "+" or word == "/" or word == "-" or word == "?" or word == "/" or word == "\\" or word == "@" or word == "=" or word == ">" or word == "<" or word == "\'":
                word = ""
            elif len(word) == 1:
                if ord(word) >= 65 and ord(word) <= 90:
                    word = ""
                elif ord(word) >= 97 and ord(word) <= 122:
                    word = ""
                elif ord(word) >= 48 and ord(word) <= 57:
                    word = ""
                elif word == "\n":
                    word = ""
                else:
                    word = ""
            elif word in word_array:
                for j in range(len(word_array)):
                    if word_array[j] == word:
                        freq_array[j] = freq_array[j] + 1
                        word = ""
                        break
            else:
                word_array.append(word)
                freq_array.append(1)
                word = ""
        else:
            word = word + string[i]

    # print(len(word_array))
    # print(freq_array)
    fr.close()
    return word_array,freq_array

# The following functions checks for the words belonging to the x_test file and compares with them to the words with
# the vocabulary of the x_train. It then takes the count of the word common to the file and x_train vocabulary.
def finding_count(vocab,word_array,freq_array):
    content = [0 for j in range(len(vocab))]
    for i in range(len(word_array)):
        for k in range(len(vocab)):
            if word_array[i]==vocab[k]:
                content[k]=freq_array[i]
            else:
                continue
    return content


def cleaning_file(file_path,x):
    word_array,freq_array=creating_words(file_path)
    # print(x[0])
    count=finding_count(x[0],word_array,freq_array)
    return count

# print("Working Main function")
# Code Starts Here.........................
x_train,y_train=load()                # Working on the x_train/Getting  documents into a 2-D array form
print(x_train.shape)
right,wrong=0,0
# The following Code tests on 40 Random files from each group
for i in range(len(y_train)):
    pathname = os.path.dirname(sys.argv[0])
    path_temp = os.path.abspath(pathname)
    path_main = path_temp+'\\test_data'
    path_main = path_main + '\\'
    path_main= path_main + y_train[i]
    os.chdir(path_main)
    path_main = path_main + '\\'
    files = os.listdir()
    for j in range(len(files)):
        x_test_file_path=path_main+str(files[j])
        count=cleaning_file(x_test_file_path,x_train)
        y_pred=naive_bayes(x_train,y_train,count)   # Calling the multinomial naive bayes
        print(y_pred,'Predicted Y for the File Path using own code......................')
        print(y_train[i], 'Actual Y for the File Path......................')
        if (y_pred==y_train[i]):
            right=right+1
        else:
            wrong=wrong+1
print("No. of right and wrong Predictions:",right,"     ",wrong)


