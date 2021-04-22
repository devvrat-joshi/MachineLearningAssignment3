import pandas as pd
import matplotlib.pyplot as plt
from metrics import accuracy
from LogisticRegression.logistic import LogisticRegression
from metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
scalar = MinMaxScaler()
np.random.seed(42)
from sklearn.datasets import load_digits

def oneHot(y,n):
    yy = np.zeros((len(y),n))
    for i,num in enumerate(y):
        yy[i,num] = 1
    return yy

X = load_digits().data
y = load_digits().target

scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
yy = y.copy()
y = oneHot(y,10)

print("######### Q3 Part A")
LR = LogisticRegression(lr=0.1,epoch=100)
LR.fit_multiclass(X,y)
y_hat = LR.predict_multi(X)
print(accuracy(y_hat,np.argmax(y,axis=1)))


print("######### Q3 Part B")
LR = LogisticRegression(lr=0.1,epoch=100)
LR.fit_multiclass_autograd(X,y)
y_hat = LR.predict_multi(X)
print("Accuracy: "+str(accuracy(y_hat,np.argmax(y,axis=1))))

print("######### Q3 Part C")
acc=0
data = load_digits()
X = data.data
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
y = data.target
X = np.append(X, np.matrix(y).T, axis=1)
k_fold = KFold(4, shuffle=True, random_state=1)
for train, test in k_fold.split(X):
    LR = LogisticRegression(lr=0.1,epoch=100)
    train_all = X[train]
    trainX = train_all[:,:-1]
    trainy = oneHot(np.array(train_all[:,-1].T,dtype="int")[0],10).astype("int")
    test_all = X[test]
    testX = test_all[:,:-1]
    testy = oneHot(np.array(test_all[:,-1].T,dtype="int")[0],10).astype("int")
    LR.fit_multiclass(trainX,trainy)
    y_hat = LR.predict_multi(testX)
    acc += accuracy(y_hat,np.argmax(testy,axis=1))
print("Average accuracy: "+str(acc/4))

X = pd.DataFrame(X)
X = X.drop(X.shape[1]-1, axis=1)
X = np.array(X)
yHat = LR.predict_multi(X)
mat = confusion_matrix(y, yHat)
classes = np.array([0,1,2,3,4,5,6,7,8,9])
disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=classes)
disp.plot() 
plt.show()

print("######### Q3 Part D")
from sklearn.decomposition import PCA
# Take best 2 components from PCA
pca = PCA(n_components=2)
X = pca.fit_transform(load_digits().data)
# Plotting scatter plot 
plt.scatter(X[:,0],X[:,1], c=load_digits().target, cmap="Paired")
plt.colorbar()
plt.show()