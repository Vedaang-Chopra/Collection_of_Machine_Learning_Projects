import numpy as np
import matplotlib as plt
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import pandas as pd

x_o=np.genfromtxt('G:\Development\Projects\Python Projects\Machine_Learning\Introduction to Machine Learning\diabetes-Train.csv',delimiter=',')
t_o=np.genfromtxt('G:\Development\Projects\Python Projects\Machine_Learning\Introduction to Machine Learning\diabetes-Test.csv',delimiter=',')
print(type(x_o))
# x=pd.DataFrame(x_o)
# t=pd.DataFrame(t_o)
# print(type(t))
y=(x_o[:,10])
x=x_o[:,0:10]
# print(y)
print(x_o.shape)
print(t_o.shape)
# x=pd.DataFrame(y)
# print(x)
print(x.shape)
# x.columns=x_o.feature_names
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.30)
#t=((t.iloc[:,0:9]).values).reshape(t.shape[0],11)
#print(t_x.shape)
# print(type(x_test))
print(x_train.shape)

print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
algo=LinearRegression(normalize=True)
algo.fit(x_train,y_train)
# print(algo.coef_,algo.intercept_)
y_pred=algo.predict(x_test)
# # print(y_pred)
# # print(y_test)
# score_test=algo.score(x_test,y_test)
# score_train=algo.score(x_train,y_train)
# # print(score_test,score_train)
final_y=algo.predict(t_o)
# # print(algo.score(t,final_y))
# # print(final_y)
np.savetxt('G:\Development\Projects\Python Projects\Machine_Learning\Introduction to Machine Learning\sub.csv',final_y,fmt='%0.50f')