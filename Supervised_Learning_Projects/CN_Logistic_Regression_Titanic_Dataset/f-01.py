import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
def change_sex(str):
    if str=='male':
        return 0
    else:
        return 1


def change_embarked(str):
    if str=='C':
        return 0
    elif str=='Q':
        return 1
    else :
        return 2

x_o=pd.read_csv('G:\Development\Projects\Python Projects\Machine_Learning\Logistic_Regression\Titanic_Project\\0000000000002429_training_titanic_x_y_train.csv')
t_o=pd.read_csv('G:\Development\Projects\Python Projects\Machine_Learning\Logistic_Regression\Titanic_Project\\0000000000002429_test_titanic_x_test.csv')

def load_clean(x2):
    abc=(x2.describe())
    abc_fixed = abc.reset_index().replace(
        {'25%': '25_percent', '50%': '50_percent', '75%': '75_percent'}
    ).set_index('index')
    x1=x2.copy()
    # x1=x[:,0:11]
    # y1=x[:,11]
    del x1['Name']
    del x1['Ticket']
    del x1['Cabin']
    x1['Gender']=x1.Sex.apply(change_sex)
    del x1['Sex']
    x1.Age.fillna(x1.Age.mean(),inplace=True)
    del x1['Fare']
    x1['New_Embarked']=x1.Embarked.apply(change_embarked)
    del x1['Embarked']
    x1['Survived_New']=x1['Survived']
    del x1['Survived']
    abc=(x1.describe())
    abc_fixed1 = abc.reset_index().replace(
        {'25%': '25_percent', '50%': '50_percent', '75%': '75_percent'}
    ).set_index('index')
    return x1

def load_clean1(x2):
    abc=(x2.describe())
    abc_fixed = abc.reset_index().replace(
        {'25%': '25_percent', '50%': '50_percent', '75%': '75_percent'}
    ).set_index('index')
    x1=x2.copy()
    # x1=x[:,0:11]
    # y1=x[:,11]
    del x1['Name']
    del x1['Ticket']
    del x1['Cabin']
    x1['Gender']=x1.Sex.apply(change_sex)
    del x1['Sex']
    x1.Age.fillna(x1.Age.mean(),inplace=True)
    del x1['Fare']
    x1['New_Embarked']=x1.Embarked.apply(change_embarked)
    del x1['Embarked']
    abc=(x1.describe())
    abc_fixed2 = abc.reset_index().replace(
        {'25%': '25_percent', '50%': '50_percent', '75%': '75_percent'}
    ).set_index('index')
    return x1

def prediction(x,y,t):
    lab_enc=preprocessing.LabelEncoder()
    encoded=lab_enc.fit_transform(y)
    print(x.shape,type(x))
    print(encoded.shape,type(encoded))
    print(t_o.shape,type(t_o))
    lr=LogisticRegression()
    lr.fit(x,encoded)
    y_pred=lr.predict(t)
    # print(lr.predict(x))
    # print(lr.score(x,encoded))
    # print(lr.predict(x)-encoded)
    # print(lr.predict_proba(x))
    # y_pred=lr.predict_proba(t)
    np.savetxt('G:\Development\Projects\Python Projects\Machine_Learning\Logistic_Regression\Titanic_Project\sub.csv',y_pred, fmt='%0.50f',delimiter="\n")

x=load_clean(x_o)
x=x.values
x1=x[:,0:x.shape[1]]
y=x[:,x.shape[1]-1]
t=load_clean1(t_o)
print(x1.shape)
prediction(x1[:,0:6],y,t)

