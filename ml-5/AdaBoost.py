from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def DataProcess(df):
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    
    # map label to 0 and 1
    df.loc[ df['label']==' >50K', 'label'] = 1
    df.loc[ df['label']==' <=50K', 'label'] = 0
    df.loc[ df['label']==' >50K.', 'label'] = 1
    df.loc[ df['label']==' <=50K.', 'label'] = 0

    # One-hot encoding
    df = df.join(pd.get_dummies(df.workclass, prefix = 'workclass'))
    df = df.join(pd.get_dummies(df.education, prefix = 'education') )
    df = df.join(pd.get_dummies( df['marital-status'], prefix = 'marital-status') )
    df = df.join(pd.get_dummies(df.occupation, prefix = 'occupation'  ) )
    df = df.join(pd.get_dummies(df.relationship, prefix = 'relationship' ) )
    df = df.join(pd.get_dummies(df.race, prefix = 'race' ) )
    df = df.join(pd.get_dummies(df.sex, prefix = 'sex' ) )
    df = df.join(pd.get_dummies(df['native-country'], prefix ='native-country' ) )
    df = df.drop(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex' , 'native-country'], axis = 1)
    return df

class AdaBoost:
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.base = []
        self.alpha = []
        self.alpha_sum = 0
        return
    
    def train(self, train_data, train_label):
        D = np.ones(train_label.shape[0])
        #print(train_label.shape[0])
        D = D / (train_label.shape[0])
        for _ in range(self.max_iter):
            #Tree = DecisionTreeClassifier( class_weight = "balanced", max_leaf_nodes = 2, max_features = 10, max_depth = 10)    
            Tree = DecisionTreeClassifier(max_leaf_nodes = 3)
            #Tree = DecisionTreeClassifier()
            Tree.fit(train_data, train_label, sample_weight = D)
            h = Tree.predict(train_data)
           
            difference = h - train_label
            #print("In acc = ", metrics.accuracy_score(train_label, h))
            pos = np.where(difference!=0)
           
            difference[difference != 0] = 1
            
            error = np.dot(D , difference)
            
            if(error>0.5):
                break
            if(error == 0):
                al = 1
                self.alpha_sum += al
                self.base.append(Tree)
                self.alpha.append(al)
                return 
            else:
                al = 0.5* np.log((1-error)/error)
            Z_t = 2 * math.sqrt(error *(1-error))

            difference[difference ==0] = -1
      
            difference = difference * al

            D = D * np.exp(difference)
            D = D/Z_t
            D = np.array(D)
            self.base.append(Tree)
            self.alpha.append(al)
            self.alpha_sum += al
        return
            
    def predict(self, test_data):    
        h = np.zeros(test_data.shape[0])
        for i in range(len(self.base)):
            Tree = self.base[i]
            al = self.alpha[i]
            pre = Tree.predict(test_data)
            pre[pre ==0] =-1
            h = h + pre * al
        h[h<0] = 0
        h[h>0] = 1
        return h
    
    def predict_proba(self, test_data):
        h = np.zeros(test_data.shape[0])  
        for i in range(len(self.base)):
            Tree = self.base[i]
            al = self.alpha[i]
            pre = Tree.predict_proba(test_data)
            #print(pre)
            pre = pre[:,1]
            #print(pre)
            h = h + pre * (al/self.alpha_sum)
        return h

# Data Processing
pd.set_option('display.max_columns', None)
train_df = pd.read_csv("adult.data", header = None)
train_num = train_df.shape[0]
test_df = pd.read_csv("adult.test", header = None)
test_num = test_df.shape[0]
combined_df = pd.concat([train_df, test_df], ignore_index = True)
#print(combined_df)
df = DataProcess(combined_df)
#print(df.head(3))
data = df.copy()
#data =data.drop(['label', 'fnlwgt', 'capital-gain', 'capital-loss'],axis =1)
data = data.drop( ['label'], axis =1  )
label = df['label']
#print(data)


data = data.values
label = label.values

#print(type(data))
x_train = data[0:train_num-1]
x_test = data[train_num : ]
y_train = label[0:train_num-1]
y_test = label[train_num:]

#x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.3, random_state = 5)


'''
# 5-fold
x =[]
y = []
for i in range(0,11):
    NUM = 2**i
    ada = AdaBoost(NUM)
    x.append(NUM)
    kf = KFold(n_splits = 5)
    val_AUC = 0
    val_acc = 0
    for train_index, val_index in kf.split(x_train):
        train_data, val_data = x_train[train_index], x_train[val_index]
        train_label, val_label = y_train[train_index], y_train[val_index]
        ada.train(train_data,train_label)
        val_res = ada.predict_proba(val_data)
        val_pre = ada.predict(val_data)
        val_AUC += metrics.roc_auc_score(val_label, val_res)
        val_acc += metrics.accuracy_score(val_label, val_pre)
    val_AUC /= 5
    val_acc /= 5
    y.append(val_AUC)
    print("NUM = ",NUM," Valiadtion AUC = ", val_AUC)
    print("NUM = ", NUM ," Validation acc = ", val_acc)



plt.semilogx(x,y, basex = 2)
plt.xlabel("num of learners")
plt.ylabel("Validation AUC")
plt.title("Adaboost")
plt.savefig("Adaboost.jpg")
'''


# Train and Test
ada = AdaBoost(1024)
ada.train(x_train, y_train)
pre = ada.predict(x_test)
res = ada.predict_proba(x_test)

AUC = metrics.roc_auc_score(y_test, res)
acc = metrics.accuracy_score(y_test, pre)
print("Test AUC = ",AUC)
print("Test acc = ",acc)


