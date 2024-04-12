import os
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn import model_selection


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
    val=10000
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
    path_main = 'G:\\Development\\Projects\\Python Projects\\Machine_Learning\\Naive_Bayes\\Naive_Bayes_Project\\Train_Data\\'
    path_main = path_main + class_label
    os.chdir(path_main)
    stop_words_list = stopwords.words("english")
    path_main = path_main + '\\'
    files = os.listdir()
    word_array = []
    freq_array = []
    for k in range(len(files)):
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


