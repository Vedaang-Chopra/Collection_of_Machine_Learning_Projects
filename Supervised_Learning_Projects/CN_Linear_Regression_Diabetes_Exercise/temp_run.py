import numpy as np
import pandas as pd
def test(x_test, y_test):
    m, c = fit(x_test, y_test)
    print(m, c)
    y_pred = predict(x_test, m, c)
    cost_test = cost(x_test, y_test, m, c)
    coeff_test = coeff_determination(y_pred, y_test)
    print("Cost=", cost_test, " and Score=", coeff_test)


def train(x_train, y_train):
    m, c = fit(x_train, y_train)
    print(m, c)
    y_pred = predict(x_train, m, c)
    # print(y_pred)
    cost_train = cost(x_train, y_train, m, c)
    # print(cost_train)
    coeff_train = coeff_determination(y_pred, y_train)
    print("Cost=", cost_train, " and Score=", coeff_train)
    return m,c

def coeff_determination(y_pred, y_true):
    u = y_true - y_pred
    u = u ** 2
    u = u.sum()
    v = y_true - y_true.mean()
    v = v ** 2
    v = v.sum()
    coeff = 1 - (u / v)
    return coeff


def fit(x_fit, y_fit):
    a1 = np.zeros(x_fit.shape[1])
    a2 = np.zeros(x_fit.shape[1])
    a3 = np.zeros(x_fit.shape[1])
    a5 = np.zeros(x_fit.shape[1])
    a4 = y_fit.mean()
    m = np.zeros(x_fit.shape[1])
    # a9=x_fit.mean()
    # a10=a9**2
    # a11=(x_fit*x_fit).mean()
    for i in range(x_fit.shape[1]):
        a1[i] = x_fit[:, i].mean()
        # print(a1[i])
        a1[i] = a1[i] ** 2
        # print(a1[i])
        a2[i] = (x_fit[:, i] * y_fit).mean()
        # print(a2[i])
        a3[i] = (x_fit[:, i] * x_fit[:, i]).mean()
        # print(a3[i])
        a5[i] = x_fit[:, i].mean() * y_fit.mean()
        # print(a5[i])
    for i in range(x_fit.shape[1]):
        m[i] = (a2[i] - a5[i]) / (a3[i] - a1[i])
        temp1 = (x_fit[:, i]).mean()
        c = a4 - m[i] * temp1
    return m, c


def predict(x_pred, m, c):
    y_pred = np.zeros(x_pred.shape[0])
    for i in range(x_pred.shape[0]):
        for j in range(x_pred.shape[1]):
            # print(m[j],x_pred[i][j])
            temp = m[j] * x_pred[i][j]
        y_pred[i] = temp + c
    return y_pred


def cost(x, y, m, c):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            temp = m[j] * x[i][j]
        temp = temp + c
        y[i] = y[i] - temp
        y[i] = y[i] ** 2
    return y[i].sum()

def load():
    from sklearn import datasets
    diab = datasets.load_diabetes()
    x = diab.data
    y = diab.target
    # data = np.loadtxt('data.csv', delimiter=",")
    # print(data.shape)
    # x = data[:, 0].reshape(-1, 1)
    # y = data[:, 1].reshape(-1, 1)
    # print(x.shape, y.shape)
    x_o = np.genfromtxt(
        'G:\Development\Projects\Python Projects\Machine_Learning\Linear_Regression\Diabetes_Exercise\diabetes-Train.csv', delimiter=',')
    t_o = np.genfromtxt(
        'G:\Development\Projects\Python Projects\Machine_Learning\Linear_Regression\Diabetes_Exercise\diabetes-Test.csv',     delimiter=',')
    print(type(x_o))
    # x=pd.DataFrame(x_o)
    # t=pd.DataFrame(t_o)
    # print(type(t))
    y = (x_o[:, 10])
    x = x_o[:, 0:10]

    from sklearn import model_selection
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,test_size=1)
    print(x_train.shape, type(x_train))
    # x_train=np.array([[1,2,3,4,5],[7,8,9,10,11]])
    # y_train=np.array([6,12])
    m,c=train(x_train, y_train)
    # test(x_test,y_test)
    y_pred=predict(t_o,m,c)
    np.savetxt('G:\Development\Projects\Python Projects\Machine_Learning\Linear_Regression\Diabetes_Exercise\\temp_sub.csv',y_pred, fmt='%0.50f')




load()
