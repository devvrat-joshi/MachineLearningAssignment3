from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp
from metrics import rmse
import numpy as np
from NeuralNetworks.neural import NeuralNetworkMLPClassification, NeuralNetworkMLPRegression
from sklearn.datasets import load_digits, load_boston
def oneHot(y,n):
    yy = np.zeros((len(y),n))
    for i,num in enumerate(y):
        yy[i,num] = 1
    return yy


print("##### Accuracy training set on digits classifier")
X = load_digits().data
y = load_digits().target
X = X.reshape(X.shape[0], -1) / 255
yy = y.copy()
y = oneHot(y, 10)
NN = NeuralNetworkMLPClassification(epoch=100,layerSize=[64, 128, 128,  10],activationNames=["relu","relu","identity"])
NN.fit(X,y)
NN.predict(NN.layers,X)
y = jnp.argmax(y, axis=1)
y_hat = jnp.argmax(NN.predict(NN.layers, X), axis=1)
print("Accuracy: "+str(jnp.sum(y_hat == y)/NN.samples))

print("##### Accuracy K=3 fold on digits classifier")
acc=0
data = load_digits()
y = data.target
X = np.append(X, np.matrix(y).T, axis=1)
k_fold = KFold(3)
fold = 1
for train, test in k_fold.split(X):
    NN = NeuralNetworkMLPClassification(epoch=300,layerSize=[64, 128, 128, 10],activationNames=["relu","relu","identity"])
    train_all = X[train]
    trainX = train_all[:,:-1]
    trainy = oneHot(np.array(train_all[:,-1].T,dtype="int")[0],10).astype("int")
    test_all = X[test]
    testX = test_all[:,:-1]
    testy = oneHot(np.array(test_all[:,-1].T,dtype="int")[0],10).astype("int")
    print("Fold Running: " +str(fold))
    NN.fit(trainX,trainy)
    y_hat = NN.predict(NN.layers,testX)
    fold+=1
    testy = jnp.argmax(testy, axis=1)
    predicted_class = jnp.argmax(NN.predict(NN.layers, testX), axis=1)
    acc += jnp.mean(predicted_class == testy)
    
print("Average accuracy: "+str(acc/3))

print("##### rmse on training set on boston data regressor")

data = load_boston()
scaler = MinMaxScaler()
X = data.data
scaler.fit(X)
X = scaler.transform(X)
y = data.target
NN = NeuralNetworkMLPRegression(epoch=1000,lr=0.01,layerSize=[13, 64, 128, 128, 256, 1],activationNames=["sigmoid","sigmoid","sigmoid","sigmoid","identity"])
NN.fit(X,y)
print(rmse(NN.predict(X),y))

print("##### rmse K=3 fold on boston data regressor")
fold = 0
rm=0
X = np.append(X, np.matrix(y).T, axis=1)
k_fold = KFold(3, shuffle=True, random_state=1)
for train, test in k_fold.split(X):
    NN = NeuralNetworkMLPRegression(epoch=1000,lr=0.01,layerSize=[13, 64, 128, 256, 256, 1],activationNames=["sigmoid","sigmoid","sigmoid","sigmoid","identity"])
    train_all = X[train]
    trainX = train_all[:,:-1]
    trainy = np.array(train_all[:,-1].T)[0]
    test_all = X[test]
    testX = test_all[:,:-1]
    testy = np.array(test_all[:,-1].T)[0]
    NN.fit(trainX,trainy)
    fold+=1
    print("Fold Running : " +str(fold))
    y_hat = NN.predict(testX)
    rm += rmse(y_hat,testy)
print("Average RMSE: "+str(rm/3))