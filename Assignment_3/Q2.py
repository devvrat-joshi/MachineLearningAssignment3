import numpy as np
from LogisticRegression.logistic import LogisticRegression
from metrics import *
from sklearn.preprocessing import MinMaxScaler         
from sklearn.model_selection import KFold
np.random.seed(42)

from sklearn.datasets import load_breast_cancer

X = load_breast_cancer().data
y = load_breast_cancer().target
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
print(load_breast_cancer().keys())
print("######### Q2 Part A")
print('##### Autograd L1 Regularized')
LR = LogisticRegression(lr = 0.1, epoch = 100, regularization='L1',lbda=0.01)
LR.fit_autograd(X, y)
y_hat = LR.predict(X)
print(accuracy(y_hat, y))

print('##### Autograd L2 Regularized')
LR = LogisticRegression(lr = 0.1, epoch = 100, regularization='L2',lbda=0.1)
LR.fit_autograd(X, y)
y_hat = LR.predict(X)
print(accuracy(y_hat, y))


def predict(X,coef):
    """
        Predict X (samples) with coefficent (coef)
        return predicted values
    """
    bias = np.ones((X.shape[0], 1))
    X = np.append(bias, X, axis=1)
    X = np.array(X)
    Z = 1/(1+np.exp(-(X.dot(coef))))		
    Y = np.round(Z)
    return Y


print("######### Q2 Part B")
print("Nested Cross Validation with L1 regularization")
acc=0
fold = 1
X = np.append(X, np.matrix(y).T, axis=1)
ft = load_breast_cancer().feature_names
k_fold = KFold(3)
for train, test in k_fold.split(X):
    train_all = X[train]
    trainX = train_all[:,:-1]
    trainy = np.array(train_all[:,-1].T)[0]
    test_all = X[test]
    testX = test_all[:,:-1]
    testy = np.array(test_all[:,-1].T)[0]
    trainX = np.append(trainX, np.matrix(trainy).T, axis=1)
    val_fold = KFold(3, shuffle=True, random_state=1)
    forthisFoldBestModel = []
    averageAccuracy = []
    for hyperparameter in range(1,10):
        acc = 0
        maxAccuracy = 0
        for train_val, val_val in val_fold.split(trainX):
            LR = LogisticRegression(lr=0.1,epoch=100,regularization="L1",lbda=hyperparameter/100)
            train_all_val = X[train_val]
            trainX_val = train_all_val[:,:-1]
            trainy_val = np.array(train_all_val[:,-1].T)[0]
            test_all_val = X[val_val]
            testX_val = test_all_val[:,:-1]
            testy_val = np.array(test_all_val[:,-1].T)[0]
            LR.fit_autograd(trainX_val,trainy_val)    
            y_hat = LR.predict(testX_val)
            ac = accuracy(y_hat,testy_val)
            acc+=ac
            if maxAccuracy<ac:
                maxAccuracy = ac
                bestmodel = LR.coef_
        averageAccuracy.append(acc/3)
        forthisFoldBestModel.append(bestmodel)
    ind = averageAccuracy.index(max(averageAccuracy))
    print("Fold {} Best Lambda: ".format(fold)+str((ind+1)/100))
    fold+=1
    coef = forthisFoldBestModel[ind]
    bestFeatures = sorted([(x,i) for i,x in enumerate(coef)])
    features = []
    for best in bestFeatures[-3:]:
        features.append(ft[best[1]])
    print("Best Features: ",features)
        
        
    yhat = predict(testX,coef)
    print(accuracy(yhat,testy))

print("Nested Cross Validation with L2 regularization")
fold = 1
acc=0
k_fold = KFold(3)
for train, test in k_fold.split(X):
    train_all = X[train]
    trainX = train_all[:,:-1]
    trainy = np.array(train_all[:,-1].T)[0]
    test_all = X[test]
    testX = test_all[:,:-1]
    testy = np.array(test_all[:,-1].T)[0]
    trainX = np.append(trainX, np.matrix(trainy).T, axis=1)
    val_fold = KFold(3, shuffle=True, random_state=1)
    forthisFoldBestModel = []
    averageAccuracy = []
    for hyperparameter in range(1,10):
        acc = 0
        maxAccuracy = 0
        for train_val, val_val in val_fold.split(trainX):
            LR = LogisticRegression(lr=0.1,epoch=100,regularization="L2",lbda=hyperparameter/100)
            train_all_val = X[train_val]
            trainX_val = train_all_val[:,:-1]
            trainy_val = np.array(train_all_val[:,-1].T)[0]
            test_all_val = X[val_val]
            testX_val = test_all_val[:,:-1]
            testy_val = np.array(test_all_val[:,-1].T)[0]
            LR.fit_autograd(trainX_val,trainy_val)    
            y_hat = LR.predict(testX_val)
            ac = accuracy(y_hat,testy_val)
            acc+=ac
            if maxAccuracy<ac:
                maxAccuracy = ac
                bestmodel = LR.coef_
        averageAccuracy.append(acc/3)
        forthisFoldBestModel.append(bestmodel)
    ind = averageAccuracy.index(max(averageAccuracy))
    print("Fold {} Best Lambda: ".format(fold)+str((ind+1)/100))
    fold+=1
    coef = forthisFoldBestModel[ind]
    yhat = predict(testX,coef)
    print(accuracy(yhat,testy))
