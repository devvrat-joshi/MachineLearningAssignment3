import pandas as pd                     # pandas
import matplotlib.pyplot as plt         # plotting purpose
# Import Autograd modules here
from autograd import grad
import autograd.numpy as np             # autograd numpy
from autograd import elementwise_grad   # autograd elementwise gradient
import imageio                          # GIF purpose
import os                               # remove intermediate generated png
class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.N = len(X)                                 # Number of samples
        if self.fit_intercept:                          # if bais present
            self.X = X.copy()                           # make of copy of X
            self.X.insert(0,"bias",1)                   # insert bais column
        else:
            self.X = X.copy()                           # else just copy X
        self.Xnumpy = self.X.to_numpy()                 # save X as numpy element
        self.numOfFeatures = len(self.X.columns)        # number of features
        numberOfBatchesPerEpoch = self.N//batch_size    # number of batches per epoch
        self.coef_ = [0 for i in range(len(self.X.columns))]    # thetas
        iterationNumber = 0                                     # iteration number for inverse lr
        for iteration in range(n_iter//numberOfBatchesPerEpoch):    # Here iteration = number of times update of thetas is done
            for batch in range(0,self.N,batch_size):                # for each batch
                y_hat = [0]*batch_size                              # calculate y_hat by multiplying the values from X
                for i in range(batch,min(self.N,batch+batch_size)):     
                    for j in range(self.numOfFeatures):
                        y_hat[i-batch] += self.Xnumpy[i,j]*self.coef_[j]
                err = [y[i]-y_hat[i] for i in range(batch_size)]    # calculate error
                errs = []                                           # errs stores the sum in err function
                for i in range(self.numOfFeatures):                 # for number of features
                    sigmaErr = 0
                    for j in range(batch,min(self.N,batch+batch_size)): # found the sigma i
                        sigmaErr += -self.Xnumpy[j,i]*err[j-batch]      
                    errs.append(sigmaErr)                                
                    iterationNumber+=1                              
                    if lr_type=="constant":             
                        self.coef_[i] = self.coef_[i] - 2*(lr/batch_size)*sigmaErr  # update thetas
                    elif lr_type=="inverse":
                        self.coef_[i] = self.coef_[i] - 2*((lr/iterationNumber)/batch_size)*sigmaErr


    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant',initialize = 0):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.theta_history = []                     # store thetas for each iteration
        self.N = len(X)                                 # Number of samples
        self.y = np.matrix(y.copy())                    # save as numpy matrix y
        if self.fit_intercept:                          # if bais present
            self.X = X.copy()                           # make of copy of X
            self.X.insert(0,"bias",1)                   # insert bais column
        else:
            self.X = X.copy()                           # else just copy X
        self.Xnumpy = self.X.to_numpy()                 # save X as numpy element
        self.XnumpyTranspose = self.Xnumpy.transpose()  # transpose of X
        self.numOfFeatures = len(self.X.columns)        # number of features
        numberOfBatchesPerEpoch = self.N//batch_size    # number of batches per epoch
        self.coef_ = np.matrix([initialize for i in range(len(self.X.columns))])    # thetas
        iterationNumber = 0                                     # iteration number for inverse lr
        for iteration in range(n_iter//numberOfBatchesPerEpoch):    # Here iteration = number of times update of thetas is done
            for batch in range(0,self.N,batch_size):                # for each batch
                y_hat = self.sigmoid(self.coef_*self.XnumpyTranspose[:,batch:min(self.N,batch+batch_size)])   # calculate y_hat vectorized
                err = self.y[:,batch:min(self.N,batch+batch_size)]-y_hat            # err vectorized
                sigmaErr = -err*self.Xnumpy[batch:min(self.N,batch+batch_size),:]   # sigma err vectorized
                iterationNumber+=1                                          
                if lr_type=="constant":                                     
                    self.coef_ = self.coef_ -(lr/batch_size)*sigmaErr     # update for lr
                elif lr_type=="inverse":
                    self.coef_ = self.coef_ -2*((lr/iterationNumber)/batch_size)*sigmaErr   # update for inverse lr
                self.theta_history.append(self.coef_.copy())    # store theta in history

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        
        self.N = len(X)                                 # Number of samples
        self.y = np.array(y.copy())
        if self.fit_intercept:                          # if bais present
            self.X = X.copy()                           # make of copy of X
            self.X.insert(0,"bias",1)                   # insert bais column
        else:
            self.X = X.copy()                           # else just copy X
        self.Xnumpy = np.array(self.X)                  # X to numpy
        self.numOfFeatures = len(self.X.columns)        # number of features
        self.batch_size = batch_size                    # batch size store
        numberOfBatchesPerEpoch = self.N//batch_size    # number of batches per epoch
        self.coef_ = np.zeros(self.X.shape[1])          # parameters
        iterationNumber = 0                             # inverse lr iteration number
        for iteration in range(n_iter//numberOfBatchesPerEpoch):    # Here iteration = number of times update of thetas is done
            for batch in range(0,self.N,batch_size):                # for each batch
                self.batch = batch                                  # current batch index
                gradient_ = elementwise_grad(self.cost)             # gradient function autograd
                iterationNumber+=1
                if lr_type=="constant":
                    self.coef_ -= lr*gradient_(self.coef_)          # update lr
                elif lr_type=="inverse":
                    self.coef_ -= (lr/iterationNumber)*gradient_(self.coef_) # update inverse lr
                

    def cost(self,theta):
        """
            Autograd cost function
            theta: learning parameters
        """
        return ((self.y[self.batch:min(self.N,self.batch+self.batch_size)] - np.dot(self.Xnumpy[self.batch:min(self.N,self.batch+self.batch_size),:],theta))**2)/len(self.X)    

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''        
        self.N = len(X)                                 # Number of samples
        self.y = np.matrix(y.copy())                    # save as numpy matrix y
        if self.fit_intercept:                          # if bais present
            self.X = X.copy()                           # make of copy of X
            self.X.insert(0,"bias",1)                   # insert bais column
        else:
            self.X = X.copy()                           # else just copy X
        self.XnumpyTranspose = np.transpose(X)          # storing X transpose
        XtX = np.linalg.inv(self.XnumpyTranspose.dot(X))
        Xty = self.XnumpyTranspose.dot(y)
        self.coef_ = XtX.dot(Xty)

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if self.fit_intercept:                          # if bais present
            self.X = X.copy()                           # make of copy of X
            self.X.insert(0,"bias",1)                   # insert bais column
        else:
            self.X = X.copy()                           # else just copy X
        if type(self.coef_)==np.matrix:                            
            return (self.coef_*self.X.to_numpy().transpose()).tolist()[0]
        return list(self.sigmoid(self.X.dot(self.coef_)))

    def predict_on_demand(self,X,coef_):
        """
            predict for a given theta
            X: feature matrix
            theta: learned parameters custom
        """
        if self.fit_intercept:                          # if bais present
            self.X = X.copy()                           # make of copy of X
            self.X.insert(0,"bias",1)                   # insert bais column
        else:
            self.X = X.copy()                           # else just copy X
        if type(coef_)==np.matrix:
            return (coef_*self.X.to_numpy().transpose()).tolist()[0]
        return list(self.X.dot(coef_))

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        # making meshgrid
        x = np.arange(-50, 50, 1)
        yy = np.arange(-50, 50, 1)
        theta1, theta0 = np.meshgrid(x, yy)
        y = y.tolist()
        N = len(X)
        # generating the output of cost function with data points
        datapoints = [[X[i],y[i]] for i in range(len(X))]
        z = sum([((theta1*datapoints[i][0] + theta0 - datapoints[i][1])**2) for i in range(N)])/N  # x^2+y^2 
        files = []
        fig = plt.figure()      # creates space for a figure to be drawn 
        # 3D plotting
        axes = fig.gca(projection ='3d')
        axes.plot_surface(theta1, theta0, z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5) 
        axes.view_init(45,30)
        axes.set_xlabel('theta1')
        axes.set_ylabel('theta0')
        axes.set_zlabel('Cost')
        plt.title("Surface plot animated GIF gradient descent")
        # for each theta, make a point on plot and save as png
        for k in range(0,100,10):
            t0,t1 = t_0[k],t_1[k]
            axes.plot([t1],[t0],[100+sum([((t1*datapoints[i][0] + t0 - datapoints[i][1])**2) for i in range(N)])/N], marker = '*', color = 'g', alpha = 1, label = 'Gradient descent')
            files.append("./temp/{}.png".format(str(k)))
            plt.savefig("./temp/{}.png".format(str(k)))
        
        # make a gif from images
        with imageio.get_writer('q7_surface.gif', mode='I') as writer:
            for fileName in files:
                for j in range(5):
                    image = imageio.imread(fileName)
                    writer.append_data(image)

        for i in set(files):        # remove all the files
            os.remove(i)
        plt.close()

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        files = []      # store pngs
        # for each theta values, plot a line and making a gif
        for i in range(0,300,20):
            plt.close()
            plt.axis([min(X)-1,max(X)+1,min(y)-4,max(y)+1])
            plt.scatter(X, y, color = "red")
            plt.plot(X, self.predict_on_demand(pd.DataFrame(X),self.theta_history[i]), color = "blue")
            plt.title("X vs y line plot")
            plt.xlabel("X")
            plt.ylabel("y")
            files.append("./temp/{}.png".format(i))
            plt.savefig("./temp/{}.png".format(i))

        # make a gif from images
        with imageio.get_writer('q7_line.gif', mode='I') as writer:
            for fileName in files:
                for j in range(5):
                    image = imageio.imread(fileName)
                    writer.append_data(image)

        for i in set(files):        # remove all the files
            os.remove(i)
        plt.close()

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        # making meshgrid
        x = np.arange(-60, 60, 1)
        yy = np.arange(-60, 60, 1)
        N = len(X)
        y = y.tolist()
        plt.close()
        theta1, theta0 = np.meshgrid(x, yy)
        # generating the output of cost function with data points
        datapoints = [[X[i],y[i]] for i in range(N)]
        z = sum([((theta1*datapoints[i][0] + theta0 - datapoints[i][1])**2) for i in range(N)])/N  # x^2+y^2 
        plt.contour(theta1,theta0,z,70,cmap="jet")
        files = []
        plt.xlabel("theta1")
        plt.ylabel("theta0")
        plt.title("Contour plot animated GIF gradient descent")
        # scatter points on contour and save pngs
        for k in range(0,100,10):
            t0,t1 = t_0[k],t_1[k]
            plt.scatter([t1],[t0],s=30, color='r', alpha=1, label='Gradient descent')
            files.append("./temp/{}.png".format(str(k)))
            plt.savefig("./temp/{}.png".format(str(k)))

        # make a gif from images
        with imageio.get_writer('q7_contour.gif', mode='I') as writer:
            for fileName in files:
                for j in range(5):
                    image = imageio.imread(fileName)
                    writer.append_data(image)

        for i in set(files):        # remove all the files
            os.remove(i)
        plt.close()
    
    def sigmoid(z):
        return 1/(1 + np.exp(-z))