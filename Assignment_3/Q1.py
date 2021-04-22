from metrics import accuracy
from LogisticRegression.logistic import LogisticRegression
from metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

np.random.seed(42)

""" Dataset Preprocessing """
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
scalar = MinMaxScaler()
scalar.fit(X)

X = scalar.transform(X)
print("######### Q1 Part A")
LR = LogisticRegression(lr=0.1,epoch=1000)
LR.fit(X,y)
ans = 0
y_hat = LR.predict(X)
print("Accuracy: ",end = "")
print(accuracy(y_hat,y))

print("######### Q1 Part B")
LR = LogisticRegression(lr=0.1,epoch=1000)
LR.fit_autograd(X,y)
ans = 0
y_hat = LR.predict(X)
print("Accuracy: ",end = "")
print(accuracy(y_hat,y))

print("######### Q1 Part C")
acc=0
X = np.append(X, np.matrix(y).T, axis=1)
k_fold = KFold(3, shuffle=True, random_state=1)
for train, test in k_fold.split(X):
    LR = LogisticRegression(lr=0.1,epoch=1000)
    train_all = X[train]
    trainX = train_all[:,:-1]
    trainy = np.array(train_all[:,-1].T)[0]
    test_all = X[test]
    testX = test_all[:,:-1]
    testy = np.array(test_all[:,-1].T)[0]
    LR.fit(trainX,trainy)
    ans = 0
    y_hat = LR.predict(testX)
    for i in range(len(testy)):
        if(testy[i]==y_hat[i]):
            ans+=1
    acc += accuracy(y_hat,testy)
print("K=3 Average Accuracy: ",end = "")
print(acc/3)

print("######### Q1 Part D")
print("See Plot")
X = load_breast_cancer().data
features = [12,14]
X = X[:,features]
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
LR = LogisticRegression(lr=0.1,epoch=100)
LR.fit(X,y)
LR.plot_decision_boundary(X,y)