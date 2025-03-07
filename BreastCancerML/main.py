#package import
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# importing ML methods
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    x = scaler.fit_transform(x)
    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return data, x, y

def split(df):
    train, valid, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    return train, valid, test

def TT(train, test):
    train, X_train, Y_train = scale_dataset(train, oversample=True)
    test, X_test, Y_test = scale_dataset(test, oversample=False)

    return train, X_train, Y_train, test, X_test, Y_test

def TVT(train, valid, test):
    train, X_train, Y_train = scale_dataset(train, oversample=True)
    valid, X_valid, Y_valid = scale_dataset(valid, oversample=False)
    test, X_test, Y_test = scale_dataset(test, oversample=False)

    return train, X_train, Y_train, valid, X_valid, Y_valid, test, X_test, Y_test

def kNN(train, X_train, Y_train, test, X_test, Y_test):

    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(X_train, Y_train)
    Y_pred = knn_model.predict(X_test)
    print("K-Nearest Neighbor:  \n" + classification_report(Y_test, Y_pred))

def naiveBayes(train, X_train, Y_train, test, X_test, Y_test):
    nb_model = GaussianNB()
    nb_model.fit(X_train, Y_train)
    Y_pred = nb_model.predict(X_test)
    print("Naive Bayes:  \n" + classification_report(Y_test, Y_pred))

def logReg(train, X_train, Y_train, test, X_test, Y_test):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, Y_train)
    Y_pred = lr_model.predict(X_test)
    print("Logistic Regression:  \n" + classification_report(Y_test, Y_pred))

def svc(train, X_train, Y_train, test, X_test, Y_test):
    svc_model = SVC()
    svc_model.fit(X_train, Y_train)
    Y_pred = svc_model.predict(X_test)
    print("Support Vector Machines:  \n" + classification_report(Y_test, Y_pred))

if __name__ == '__main__':
    cols = ["ID", "cThick", "UCSize", "UCShape", "Adhesion", "CECSize", "Bare", "Bland", "Normal", "Mitoses", "class"]
    df = pd.read_csv("./breast-cancer-wisconsin.data", names=cols)
    df.drop("Bare", inplace=True, axis=1)

    train, valid, test = split(df)

    # dummy variables for the splitting, done to assure consistency among all ML algorithms
    a, b, c, d, e, f = TT(train, test)
    kNN(a,b,c,d,e,f)
    naiveBayes(a,b,c,d,e,f)
    logReg(a,b,c,d,e,f)
    svc(a,b,c,d,e,f)