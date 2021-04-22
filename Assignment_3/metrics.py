import math
import pandas as pd
import numpy as np
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    if type(y_hat)==pd.core.series.Series:
        y_hat = y_hat.tolist()
    if type(y)==pd.core.series.Series:
        y = y.tolist()
    ans = 0
    for i in range(len(y)):
        if y[i]==y_hat[i]:
            ans+=1
    return ans/len(y)
    # TODO: Write here

def precision(y_hat, y, cls):
    """
    Function to calculate the precision
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    if type(y_hat)==pd.core.series.Series or type(y_hat)==np.ndarray:
        y_hat = y_hat.tolist()
        print("ASDF")
    if type(y)==pd.core.series.Series  or type(y)==np.ndarray:
        y = y.tolist()
    ans = 0
    l = len(y)
    for i in range(l):
        if y_hat[i]==cls and y[i]==cls:
            ans+=1
    if y_hat.count(cls)==0:
        return 1
    return ans/y_hat.count(cls)

def recall(y_hat, y, cls):
    """
    Function to calculate the recall
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    if type(y_hat)==pd.core.series.Series or type(y)==np.ndarray:
        y_hat = y_hat.tolist()
    if type(y)==pd.core.series.Series or type(y)==np.ndarray:
        y = y.tolist()
    ans = 0
    l = len(y)
    for i in range(l):
        if y[i]==cls and y_hat[i]==cls:
            ans+=1
    if y.count(cls)==0:
        return 1
    return ans/y.count(cls)

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    if type(y_hat)==pd.core.series.Series:
        y_hat = y_hat.tolist()
    if type(y)==pd.core.series.Series:
        y = y.tolist()
    ans = 0
    for i in range(len(y_hat)):
        ans += (y_hat[i]-y[i])**2
    return (ans/len(y_hat))**(1/2)

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    if type(y_hat)==pd.core.series.Series:
        y_hat = y_hat.tolist()
    if type(y)==pd.core.series.Series:
        y = y.tolist()
    ans = 0
    for i in range(len(y_hat)):
        ans += abs(y_hat[i]-y[i])
    return ans/len(y)
