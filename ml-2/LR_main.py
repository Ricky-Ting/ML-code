import pandas as pd
import numpy as np
from numpy import *
def sigmoid(X):
    return (1.0 / (1 + exp(-X)))
def list_add(A,B):
    ret = []
    for i in range(len(A)):
        ret.append(A[i]+B[i])
    return ret
def list_div(A,B):
    ret = []
    for i in range(len(A)):
        if(B[i]!=0):
            ret.append(A[i]/B[i])
        else:
            ret.append(1)
    return ret
def list_mean(A):
    sum = 0;
    for i in range(len(A)):
        sum += A[i]
    return sum/len(A)

class Logistic_Regression:
    def __init__(self, max_iter):
        self.max_iter = max_iter;
        return


    def train(self, alpha, X, y):
        self.w = np.zeros(X.shape[1])
        self.alpha = alpha
        GD = np.zeros(X.shape[1])
        predicted_y = np.zeros(X.shape[0])
        error = np.zeros(X.shape[0])
        for _ in range(self.max_iter):
            #print(X.shape," ",self.w.shape)
            predicted_y = sigmoid(np.dot(X , self.w))
            #print(y.shape," ", predicted_y.shape)
            error = y - predicted_y
            GD = np.dot(X.transpose() ,  error) / X.shape[0]
            #print(GD.shape)
            self.w = self.w + self.alpha * GD
        return

    def predict(self, X):
        return np.dot(X , self.w)

        print(len(TP), len(FP), len(FN), len(TN))
class Multi_class:
    def __init__(self, class_num, max_iter, X, Y):
        self.class_num = class_num
        self.X = X
        self.Y = Y
        self.model = []
        self.max_iter = max_iter
        for i in range(self.class_num):
           self.model.append(Logistic_Regression(self.max_iter)) 
        return

    def train(self, alpha):
        self.alpha = alpha
        for i in range(self.class_num):
            y = np.copy(self.Y)
            for j in range(len(y)):
                if(y[j]==i+1):
                    y[j] = 1
                else:
                    y[j] = 0
            self.model[i].train(self.alpha, self.X, y )
        return

    def predict(self, X):
        #print(X)
        y = np.zeros(X.shape[0])
        for i in range(len(X)):
            predicted = []
            for j in range(self.class_num):
                predicted.append(self.model[j].predict(X.iloc[i]))
            y[i] = (predicted.index(max(predicted)) + 1)
        return y
    
    def evaluate(self, X, Y):
        TP = []
        FP = []
        FN = []
        TN = []
        for i in range(self.class_num):
            TP_counter = 0
            FP_counter = 0
            FN_counter = 0
            TN_counter = 0
            y = self.model[i].predict(X)
            for j in range(len(Y)):
                if(Y[j]==i+1 and y[j]>=0.5):
                    TP_counter = TP_counter + 1
                if(Y[j]!=i+1 and y[j]>=0.5):
                    FP_counter = FP_counter + 1
                if(Y[j]==i+1 and y[j]<0.5):
                    FN_counter = FN_counter + 1
                if(Y[j]!=i+1 and y[j]<0.5):
                    TN_counter = TN_counter + 1
            if(TP_counter + FN_counter == 0):
                print(i+1)
            TP.append(TP_counter)
            FP.append(FP_counter)
            FN.append(FN_counter)
            TN.append(TN_counter)
        #print(TP)
        #print(FP)
        #print(FN)
        #print(TN)
        P = list_div(TP, list_add(TP,FP) )
        R = list_div(TP, list_add(TP,FN) )
        macro_P = list_mean(P)
        macro_R = list_mean(R)
        macro_F1 = (2 * macro_P * macro_R) / (macro_P + macro_R)
        micro_P = list_mean(TP) / (list_mean(TP) + list_mean(FP))
        micro_R = list_mean(TP) / (list_mean(TP) + list_mean(FN))
        micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R)
        print("micro Precission = ", micro_P)
        print("micro Recall = ", micro_R)
        print("micro F1 = ", micro_F1)
        print("macro Precission = ", macro_P)
        print("macro Recall = ", macro_R)
        print("macro F1 = ", macro_F1)
        return


def accuracy(y1, y2):
    counter = 0
    for i in range(len(y1)):
        if(y1[i] == y2[i]):
            counter = counter + 1
    return counter/len(y1)
def Standardlization(train, test):
    for i in range(train.shape[1]):
        mymean = train.iloc[:, i].mean()
        mystd = train.iloc[:, i].std()
        train.iloc[:, i] = (train.iloc[:, i] - mymean) / ( mystd )
        test.iloc[:, i]= (test.iloc[:, i] - mymean) / ( mystd)
    return [train, test]

    
if __name__=="__main__":
    df_train = pd.read_csv("train_set.csv")
    df_test = pd.read_csv("test_set.csv")
    #print(df.shape)
    #print(df_train.iloc[0])
    X_train = df_train.iloc[ : , 0:16 ]
    Y_train = df_train.iloc[ : , 16]

    X_test = df_test.iloc[ : , 0:16 ]
    Y_test = df_test.iloc[ : , 16]

    [X_train, X_test] = Standardlization(X_train, X_test)
    X_train.insert(16, "b", 1)
    X_test.insert(16, "b", 1)
    #print(X_train)
    #X_train = np.c_[X_train, np.ones(X_train.shape[0])]
    #X_test = np.c_[X_test, np.ones(X_test.shape[0])]
    #print(X_train[0])
    #print(X_train.iloc[0])
    '''
    max_acc = 0
    best_alpha = 0
    best_iter = 0
    for j in range(5,15): 
        for i in range(1,10):
            model = Multi_class(26, j*500, X_train, Y_train)
            model.train(i/10)
            results = model.predict(X_test)
            acc = accuracy(Y_test, results)
            print("In iter = ", j*500,"alpha = ", i/10 )
            if(acc>max_acc):
                max_acc = acc
                best_alpha = i/10
                best_iter = j*500
                print("new improvemnt: acc = ",max_acc," iter = ",best_iter, " best_alpha = ",best_alpha )

    print(max_acc, " ", best_alpha, " ", best_iter)
    '''
    # 5000 0.05
    model = Multi_class(26,10000, X_train, Y_train)
    model.train(0.1)
    results = model.predict(X_test)
    print("accuracy = ",accuracy(Y_test, results))
    model.evaluate(X_test, Y_test)

