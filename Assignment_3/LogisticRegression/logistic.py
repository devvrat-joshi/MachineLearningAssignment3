import matplotlib.pyplot as plt         # plotting purpose
# Import Autograd modules here
import autograd.numpy as np             # autograd numpy
from autograd import elementwise_grad   # autograd elementwise gradient

class LogisticRegression:
    """
        class logistic regression
            binary and multiclass both supported
            autograd included
            L1 and L2 regularized also present in this class
    """
    def __init__(self,lr=0.1,epoch=100,regularization=None,lbda=0.1):
        """
            lr: learning rate
            epoch: number of epochs
            regularization: regularization type L1 or L2
            lbda: regularization penalty
        """
        self.lr = lr
        self.epoch = epoch
        self.regularization = regularization
        self.lbda = lbda
        
    def fit(self,X,y):
        """
            trains the model
            X: samples
            y: labels
            returns trained weights
        """
        self.X = X.copy()
        bias = np.ones((self.X.shape[0],1)) 
        self.X = np.append(bias,self.X,axis=1)   # Include Bais
        self.X = np.array(self.X)               
        self.y = y
        self.nofFeatures = len(self.X[0])
        self.samples = len(self.X)
        self.coef_= np.ones(self.nofFeatures)    # init features
        for i in range(self.epoch):
            err = self.sigmoid(self.X.dot(self.coef_))-y        # error
            self.coef_= self.coef_- self.lr*(err).dot(self.X) # coef
        return self.coef_
    
    def predict(self,X):
        """
            predicts according to the trained model
            X: samples
            return prediction
        """
        self.X = X
        bias = np.ones((self.X.shape[0],1))
        self.X = np.append(bias,self.X,axis=1)
        self.X = np.array(self.X)
        Z = 1/(1+np.exp(-(self.X.dot(self.coef_ ))))		
        Y = np.round(Z)
        return Y
    
    def fit_autograd(self,X,y):
        """
            fit with autograd
            X: samples
            y: labels
            returns trained weights
        """
        self.X = X
        bias = np.ones((self.X.shape[0],1))
        self.X = np.append(bias,self.X,axis=1)
        self.X = np.array(self.X)
        self.y = y
        self.nofFeatures = len(self.X[0])
        self.samples = len(self.X)
        self.coef_= np.random.randn(self.nofFeatures)
        if self.regularization==None:
            gradient_ = elementwise_grad(self.cost)
        if self.regularization=="L1":
            gradient_ = elementwise_grad(self.costL1)
        if self.regularization=="L2":
            gradient_ = elementwise_grad(self.costL2)
        for i in range(self.epoch):
            self.coef_ -= self.lr*gradient_(self.coef_)
        return self.coef_
    
    def cost(self,theta):
        """
            returns cross entropy loss on input parameters
        """
        return -np.sum((self.y*np.log(1/(1 + np.exp(-np.dot(self.X,theta))))+(1-self.y)*np.log(1-1/(1 + np.exp(-np.dot(self.X,theta))))))/self.samples
    
    def costL2(self,theta):
        """
            returns cross entropy loss on input parameters
            L2 regularized
        """
        return -np.sum((self.y*np.log(1/(1 + np.exp(-np.dot(self.X,theta))))+(1-self.y)*np.log(1-1/(1 + np.exp(-np.dot(self.X,theta))))))/self.samples+self.lbda*theta.T*theta

    def costL1(self,theta):
        """
            returns cross entropy loss on input parameters
            L1 regularized
        """
        return -np.sum((self.y*np.log(1/(1 + np.exp(-np.dot(self.X,theta))))+(1-self.y)*np.log(1-1/(1 + np.exp(-np.dot(self.X,theta))))))/self.samples+self.lbda*np.sum(np.abs(theta))

    
    def fit_multiclass(self,X,y):
        """
            multiclass learning
            X: samples
            y: labels
        """
        self.labels = np.unique(y)
        self.X = X.copy()
        bias = np.ones((self.X.shape[0],1))
        self.X = np.append(bias,self.X,axis=1)
        self.X = np.array(self.X)
        self.y = y
        self.nofFeatures = self.X.shape[1]
        self.samples = len(self.X)
        self.coef_= np.ones((self.nofFeatures,y.shape[1]))
        for i in range(self.epoch):
            err = 1/(1 + np.exp(-(self.X.dot(self.coef_))))-y
            x = sum([np.exp(self.X.dot(self.coef_[:,j])) for j in range(self.y.shape[1])])
            for j in range(self.y.shape[1]):
                err = np.exp(self.X.dot(self.coef_[:,j]))/x-self.y[:,j]
                self.coef_[:,j]= self.coef_[:,j] - self.lr*err.dot(self.X)/self.samples
        return self.coef_
    
    def fit_multiclass_autograd(self,X,y):
        """
            multiclass learning using autograd
            X: samples
            y: labels
        """
        self.labels = np.unique(y)
        self.X = X
        bias = np.ones((self.X.shape[0],1))
        self.X = np.append(bias,self.X,axis=1)
        self.X = np.array(self.X)
        self.y = y
        self.nofFeatures = self.X.shape[1]
        self.samples = len(self.X)
        self.coef_= np.ones((self.nofFeatures,y.shape[0]))
        gradient_ = elementwise_grad(self.costMulti)
        for i in range(self.epoch):
            print("\rautograd slow: epoch = "+str(i),end="")
            self.coef_= self.coef_ - self.lr*gradient_(self.coef_)/self.samples
        print()
        return self.coef_

    def predict_multi(self,X):
        """
            predict multiclass output
            X: samples
        """
        bias = np.ones((X.shape[0], 1))
        X = np.append(bias, X, axis=1)
        X = np.array(X)
        Z = 1/(1+np.exp(-(X.dot(self.coef_ ))))		
        Y = []
        for i in Z:
            Y.append(np.argmax(i))
        return Y
    
    def costMulti(self,theta):
        """
            Cost function multiclass
        """
        J = 0
        for i in range(self.samples):
            k = np.argmax(self.y[i])
            x = np.dot(self.X[i],theta)
            denom = 0
            denom=np.sum(np.exp(x))
            numerator = np.exp(np.dot(self.X[i],theta[:,k]))
            probab = np.log(numerator/denom)
            J+=probab
        return -J
        
    def plot(self, X, y):
        """
            plots decision boundary
            X: samples
            y: labels
        """
        minX,maxX = min(X[:,0])-0.1,max(X[:,0])+0.1
        minY,maxY = min(X[:,1])-0.1,max(X[:,1])+0.1
        xDim = np.array([minX,maxX])
        yDim = -(1/self.coef_[2])*(self.coef_[1]*xDim + self.coef_[0])
        plt.plot(xDim,yDim)
        plt.fill_between(xDim,yDim,minY,color='red',alpha=0.15)
        plt.fill_between(xDim,yDim,maxY,color='green',alpha=0.15)
        plt.xlim(minX,maxX)
        plt.ylim(minY,maxY)
        maskY0 = y==0
        maskY1 = y==1
        plt.scatter(*X[maskY0].T,s=5,color="brown")
        plt.scatter(*X[maskY1].T,s=5,color="blue")
        plt.title("Decision Boundary")
        plt.ylabel("x2") 
        plt.xlabel("x1")
        plt.savefig("plotQ1.png")

    def sigmoid(self,Z):
        """
            returns sigmoid of Z
        """
        return 1/(1 + np.exp(-Z))