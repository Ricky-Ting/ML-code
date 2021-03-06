from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def DataProcess(df):
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    #print(df.head())
    df.loc[ df['label']==' >50K', 'label'] = 1
    df.loc[ df['label']==' <=50K', 'label'] = 0
    df.loc[ df['label']==' >50K.', 'label'] = 1
    df.loc[ df['label']==' <=50K.', 'label'] = 0

    
    df = df.join(pd.get_dummies(df.workclass, prefix = 'workclass'))
    #df = df.join(pd.get_dummies( ) )
    df = df.join(pd.get_dummies(df.education, prefix = 'education') )
    df = df.join(pd.get_dummies( df['marital-status'], prefix = 'marital-status') )
    df = df.join(pd.get_dummies(df.occupation, prefix = 'occupation'  ) )
    df = df.join(pd.get_dummies(df.relationship, prefix = 'relationship' ) )
    df = df.join(pd.get_dummies(df.race, prefix = 'race' ) )
    df = df.join(pd.get_dummies(df.sex, prefix = 'sex' ) )
    df = df.join(pd.get_dummies(df['native-country'], prefix ='native-country' ) )
    df = df.drop(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex' , 'native-country'], axis = 1)

    return df


class RandomForest:
    def __init__(self, n_estimators ):
        self.n_estimators = n_estimators
        self.base = []

    def get_bootstrap_data(self, X, Y):
        m = X.shape[0]
        Y = Y.reshape(m,1)
        X_Y = np.hstack((X,Y))
        np.random.shuffle(X_Y)
        data_sets = []
        for _ in range(self.n_estimators):
            idm = np.random.choice(m,m,replace=True)
            bootstrap_X_Y = X_Y[idm,:]
            bootstrap_X = bootstrap_X_Y[:,:-1]
            bootstrap_Y = bootstrap_X_Y[:,-1:]
            data_sets.append([bootstrap_X, bootstrap_Y])
        return data_sets

    def train(self, X,Y):
        sub_sets = self.get_bootstrap_data(X,Y)

        for i in range(self.n_estimators):
            sub_X, sub_Y = sub_sets[i]
            Tree = DecisionTreeClassifier(max_features = "log2")
            Tree.fit(sub_X, sub_Y)
            self.base.append(Tree) 

        return

    def predict(self,X):
        h = np.zeros(X.shape[0])

        for i in range(self.n_estimators):
            h += self.base[i].predict(X)
        h = h/self.n_estimators
        h[h<0.5] = 0
        h[h>=0.5] = 1
        return h

    def predict_proba(self, X):
        h = np.zeros(X.shape[0])

        for i in range(self.n_estimators):
            h += self.base[i].predict(X)
        h = h/self.n_estimators
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


'''
# 5-fold 
x =[]
y = []
for i in range(0,11):
    NUM = 2**i
    RF = RandomForest(NUM)
    x.append(NUM)
    kf = KFold(n_splits = 5)
    val_AUC = 0
    val_acc = 0
    for train_index, val_index in kf.split(x_train):
        train_data, val_data = x_train[train_index], x_train[val_index]
        train_label, val_label = y_train[train_index], y_train[val_index]
        RF.train(train_data,train_label)
        val_res = RF.predict_proba(val_data)
        val_pre = RF.predict(val_data)
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
plt.title("RandomForest")
plt.savefig("RandomForest.jpg")

'''

# Test

RF = RandomForest(1024)
RF.train(x_train, y_train)
res = RF.predict_proba(x_test)
pre = RF.predict(x_test)

AUC = metrics.roc_auc_score(y_test, res)
acc = metrics.accuracy_score(y_test, pre)
print("Test AUC = ", AUC)
print("Test acc = ", acc)




