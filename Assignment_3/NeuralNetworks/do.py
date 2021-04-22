import numpy.random as npr
import jax
import numpy as np
from jax import jit, grad
import jax.numpy as jnp
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from functools import partial
from jax.scipy.special import logsumexp
class NeuralNetwork():
  def __init__(self, layerSize, activationNames):
    self.layerSize = layerSize.copy()
    self.activationFuncs = activationNames.copy()
    self.num_layers = len(layerSize)
    for i in range(len(self.activationFuncs)):
        if self.activationFuncs[i]=="sigmoid":
            self.activationFuncs[i] = self.sigmoid
        elif self.activationFuncs[i]=="relu":
            self.activationFuncs[i] = self.relu
        elif self.activationFuncs[i]=="identity":
            self.activationFuncs[i] = self.identity
  
  def cost_reg(self, layers, batch):
    X_cur, y_cur = batch
    n = len(X_cur)
    y_cur = y_cur.reshape(-1)
    self.layers = layers
    y_hat = self.predict( X_cur)
    print(y_hat.shape)
    return jnp.dot((y_cur-y_hat).T,(y_cur-y_hat))/n
  
  @partial(jit, static_argnums=(0,))
  def __update(self, layers, batch):
    grads = grad(self.cost_reg)(layers, batch)
    for i in range(len(self.layers)):
        layers[i][0] -= self.lr * grads[i][0]
        layers[i][1] -= self.lr * grads[i][1]
    return layers
  
  def fit(self, X, y, batch, lr = 0.01, epochs = 100):
    self.lr = lr
    self.X = X
    self.y = y.reshape(-1,1)
    input_size = len(X[0])
    layer_sizes = self.layerSize
    self.layers = []
    for i in range(1,len(self.layerSize)):
        self.layers.append([rng.randn(self.layerSize[i],self.layerSize[i-1]),rng.randn(self.layerSize[i])])
    for epoch in range(epochs):
        for i in range(0,X.shape[0],batch):
          self.layers = self.__update(self.layers, (self.X[i:i+batch,:],self.y[i:i+batch]))

  
  def predict(self, X):
    activations = X
    for i,(w, b) in enumerate(self.layers):
        activations = self.activationFuncs[i](jnp.dot(activations, w.T) + b)
    return activations.reshape(-1)
  
  def relu(self,inp):
      return jnp.maximum(inp,0)

  def sigmoid(self,inp):
      return jax.nn.sigmoid(inp)
  
  def identity(self,inp):
      return inp
  

def rmse(y, y_hat):
  n = len(y)
  rmse = 0
  for i in range(n):
    rmse += pow(y_hat[i]-y[i],2)
  rmse = pow(rmse/y.size,0.5)
  return rmse
# K fold Cross Validation, Regression case on boston Dataset.

a = [13, 64, 128, 128, 64, 1]
b = ["sigmoid","sigmoid","sigmoid","sigmoid","identity"]
data = load_boston()
scaler = MinMaxScaler()
scaler.fit(data["data"])
X = scaler.transform(data["data"])
y = data["target"]
NN = NeuralNetwork(layerSize = a, activationNames = b)
NN.fit(X,y,50, epochs = 1000, lr = 0.001)
y_hat = NN.predict(X)
rmse(y,y_hat)