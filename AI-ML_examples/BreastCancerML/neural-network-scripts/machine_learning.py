from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def knn(X_train, Y_train, X_test, Y_test, neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=neighbors)
    knn_model.fit(X_train, Y_train)
    Y_pred = knn_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

def NB(X_train, Y_train, X_test, Y_test):
    nb_model = GaussianNB()
    nb_model.fit(X_train, Y_train)
    Y_pred = nb_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

def logreg(X_train, Y_train, X_test, Y_test):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, Y_train)
    Y_pred = lr_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

def svc(X_train, Y_train, X_test, Y_test):
    svc_model = SVC()
    svc_model.fit(X_train, Y_train)
    Y_pred = svc_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))