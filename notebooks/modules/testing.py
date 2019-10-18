'''
Created on Sep 14, 2019

@author: dan
'''
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm # Time Series Analysis
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import prettytable
import matplotlib.pylab as plt
import seaborn as sns
import sys
import os
import copy
plt.style.use('ggplot')

from sklearn.metrics import confusion_matrix
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


def run_ad_fuller(X):
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] > 0.05:
        print('\nWe fail to reject the Null Hypothesis (H0) -- the time series has a unit root and is not stationary')
    else:
        print('\nWe reject the Null Hypothesis (H0) -- the time series is stationary')


def measure_error(actual, pred, label=None, show=True):
    sq_error = mean_squared_error(actual, pred)
    mse = np.mean(sq_error)**.5
    avg = np.mean(actual)
    errors = {'label':label,'RMSE': mse, 'RMSE_pcent': mse / abs(avg) }
    if show:
        pt = prettytable.PrettyTable(['metric','value'])
        for (k,v) in errors.items():
            pt.add_row([k,v])
        print(pt)
    return errors

def plot_confusion_matrix(actual, pred, show=True):
    y_true = []
    y_pred = []

    last = actual[0]
    for (a,p) in list(zip(actual, pred))[1:]:
        y_true += [-1 + int(a - last > 0) * 2]
        y_pred += [-1 + int(p - last > 0) * 2]
        last = a
    labels = ['tn', 'fp', 'fn', 'tp']
    cm = confusion_matrix(y_true, y_pred)
    
    print('Confusion Matrix\nRaw')
    print(cm)
    print('Normalized')
    cm_norm = cm / np.sum(cm)
    
    if show:
        print(cm_norm)
        plt.figure()
        plt.imshow(cm,  cmap=plt.cm.Blues)
        plt.colorbar()
        #tick_marks = np.arange(2)
        plt.xticks([-0.5,0.5], ['negative','positive'])
        plt.yticks([-0.5,0.5], ['negative','positive'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Manual AR(p) Model Predicted vs. Actual')
    return {'cm': cm, 'cm_norm': cm_norm}

def eval_plot(X, Y, Y_hat, lags=None):
    R = np.array(Y) - np.array(Y_hat)
    f, ax = plt.subplots(2, 2)
    res = stats.probplot(R, plot=ax[0,0])
    ax[0, 0].set_title('Normal Probability Plot of the Residuals')
    ax[0, 1].scatter(X,R)
    ax[0, 1].set_title('Residuals vs Fitted Values')
    ax[1, 0].hist(R)
    ax[1, 0].set_title('Histogram of the Residuals')
    ax[1, 1].plot(R)
    ax[1, 1].set_title('Residuals vs Order of the Data')
    plt.show()
    if lags is None:
        lags = min(20, len(R) / 2)
    (lb, p_values) = acorr_ljungbox(R, lags=lags, boxpierce=False)
    print('Ljung-Box Test')
    print("H_0 (p>0.05) --> The data are independently distributed -- i.e. there's no auto correlations")
    print("H_a (p<0.05) --> The data are not independently distributed -- i.e. there is auto correlations") 
    print('p_values',p_values)
    sub = list(filter(lambda p: p<.05, p_values))
    if len(sub) > 0:
        print('PROBLEM!  There appears to be information left in the residuals')
    else:
        print('There does not appear to be information left in the residuals')
    return len(sub)>0


# modified from 
# http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016

def ts_plot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        stats.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    run_ad_fuller(y)
    return 

