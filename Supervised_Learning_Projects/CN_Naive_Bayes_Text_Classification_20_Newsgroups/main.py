import naive_bayes_code as nbs
import numpy as np
import first_file as ff
from nltk.corpus import stopwords
import os
import random
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
first_file_run=ff.load
x_train,y_train=first_file_run()                # Working on the x_train/Getting  documents into a 2-D array form
right,wrong=0,0
# The following Code tests on 40 Random files from each group
for i in range(40):
    path_main = 'G:\\Development\\Projects\\Python Projects\\Machine_Learning\\Naive_Bayes\\Naive_Bayes_Project\\test_data'
    path_main = path_main + '\\'
    path_main = path_main + y_train[i]
    os.chdir(path_main)
    path_main = path_main + '\\'
    files = os.listdir()
    no=random.randint(0, len(files)-1)  # Randomizing file selection
    x_test_file_path=path_main+str(files[no])
    count=cleaning_file(x_test_file_path,x_train)
    y_pred=nbs.naive_bayes(x_train,y_train,count)   # Calling the multinomial naive bayes
    print(y_pred,'Predicted Y for the File Path......................')
    print(y_train[i], 'Actual Y for the File Path......................')
    if (y_pred==y_train[i]):
        right=right+1
    else:
        wrong=wrong+1
print("No. of right and wrong Predictions:",right,"     ",wrong)