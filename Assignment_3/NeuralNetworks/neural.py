# Referece: Jax git repo
import numpy as np          # numpy
from jax import jit, grad   # jit and grad
import jax.numpy as jnp     # jax numpy
import jax                  # jax
from functools import partial # Only to make code fast, used to remove the jit object error (stackoverflow solution, not always works to make code fast)

class NeuralNetworkMLPClassification:
    """
        Classification Class
    """
    def __init__(self,lr=0.03,epoch=100,batch=50,layerSize=[1],activationNames=["relu"]):
        """
            initialize NN
            lr: learning rate
            epoch: epoches
            batch: batch size
            layerSize: list of sizes of layers
            activationNames: names of layerwise activation functions
        """
        self.lr = lr
        self.epoch = epoch
        self.batch = batch
        self.layerSize = layerSize
        self.activiationFuncs = []
        for i in activationNames:
            if i=="relu":
                self.activiationFuncs.append(self.relu)
            elif i=="sigmoid":
                self.activiationFuncs.append(self.sigmoid)
            else:
                self.activiationFuncs.append(self.identity)
        
    def fit(self,X,y):
        """
            Backpropagation to fit NN
            X: samples
            y: labels
        """
        self.X = X
        self.y = y
        self.samples = len(X)
        self.layers = []
        for i in range(1,len(self.layerSize)):
            self.layers.append([np.random.randn(self.layerSize[i],self.layerSize[i-1]),np.random.randn(self.layerSize[i])])
        """
            stochastic gradient descent used to update all layers
        """
        for epoch in range(self.epoch):
            print("\rEpoches Completed: "+str(epoch+1),end="")
            for _ in range(0,self.samples,self.batch):
                self.layers = self.sgd(self.layers,(self.X[_:_+self.batch,:],self.y[_:_+self.batch,:]))
        print()

    def loss(self,layers):
        """
            cross entropy multiclass loss function
            layers: weights 
        """
        pred = self.predict(layers, self.batch_[0])
        return -jnp.mean(jnp.sum(pred*self.batch_[1], axis=1))
    
    @partial(jit,static_argnums=(0,))
    def sgd(self,layers,batch):
        """
            stochastic gradient descent
            layers: weights
            batch: current batch
        """
        self.batch_ = batch
        grads = grad(self.loss)(layers)
        for i in range(len(self.layers)):
            layers[i][0] -= self.lr*grads[i][0]
            layers[i][1] -= self.lr*grads[i][1]
        return layers

    def predict(self,layers,inp):
        """
            predict values / forward propagation
            layers: weights
            inp: inputs
        """
        for i, (weight, bais) in enumerate(layers):
            inp = self.activiationFuncs[i](jnp.dot(inp,weight.T)+bais)
        return jax.nn.log_softmax(inp,axis=1)        

    def relu(self,inp):
        """
        returns relu of inp
        """
        return jnp.maximum(inp,0)

    def sigmoid(self,inp):
        """
        returns sigmoid of inp
        """
        return jax.nn.sigmoid(inp)
    
    def identity(self,inp):
        """
        returns inp
        """
        return inp
    
    
class NeuralNetworkMLPRegression:
    """
        Neural Network regression
    """
    def __init__(self,lr=0.03,epoch=100,batch=50,layerSize=[1],activationNames=["relu"]):
        """
            initialize NN
            lr: learning rate
            epoch: epoches
            batch: batch size
            layerSize: list of sizes of layers
            activationNames: names of layerwise activation functions
        """
        self.lr = lr
        self.epoch = epoch
        self.batch = batch
        self.activationFuncs = activationNames
        self.layerSize = layerSize
        self.num_layers = len(layerSize)
        for i in range(len(self.activationFuncs)):
            if self.activationFuncs[i]=="sigmoid":
                self.activationFuncs[i] = self.sigmoid
            elif self.activationFuncs[i]=="relu":
                self.activationFuncs[i] = self.relu
            elif self.activationFuncs[i]=="identity":
                self.activationFuncs[i] = self.identity
    
    def fit(self,X,y):
        """
            Backpropagation to fit NN
            X: samples
            y: labels
        """
        self.X = X
        self.y = y.reshape(-1,1)
        self.layers = []
        for i in range(1,len(self.layerSize)):
            self.layers.append([np.random.randn(self.layerSize[i],self.layerSize[i-1]),np.random.randn(self.layerSize[i])])
        """
            stochastic gradient used to update weights of all layers
            Here to make the code fast, I have used the jit compiler method from jax library git repo
            It is because it takes around 1000 epochs to get a good rmse
            and code runs slow without jit support as given in jax readme
        """
        for epoch in range(self.epoch):
            print("\rEpoches Completed: "+str(epoch+1),end="")
            for i in range(0,X.shape[0],self.batch):
                self.layers = self.sgd(self.layers, (self.X[i:i+self.batch,:],self.y[i:i+self.batch]))
        print()

    def loss(self,layers):
        """
            mean squared error
            layers: weights 
        """
        y = self.batch_[1].reshape(-1)
        self.layers = layers
        pred = self.predict(self.batch_[0])
        return jnp.sum((y-pred)**2)/self.batch_[0].shape[0]
  
    @partial(jit,static_argnums=(0,))
    def sgd(self,layers,batch):
        """
            stochastic gradient descent
            layers: weights
            batch: current batch
        """
        self.batch_ = batch
        grads = grad(self.loss)(layers)
        for i in range(len(self.layers)):
            layers[i][0] -= self.lr*grads[i][0]
            layers[i][1] -= self.lr*grads[i][1]
        return layers
  
    def predict(self, X):
        """
        Forward Pass
        """
        input_ = X
        for i,(w, b) in enumerate(self.layers):
            input_ = self.activationFuncs[i](jnp.dot(input_, w.T)+b)
        return input_.reshape(-1)
  
    def relu(self,inp):
        return jnp.maximum(inp,0)

    def sigmoid(self,inp):
        return jax.nn.sigmoid(inp)
  
    def identity(self,inp):
        return inp
  