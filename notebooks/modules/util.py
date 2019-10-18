'''
Created on Oct 16, 2019

@author: dan
'''
import numpy as np
import matplotlib.pyplot as plt
import prettytable


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

def cf_matrix(actual, pred):
    cm = confusion_matrix(actual, pred)
    print('Confusion Matrix')
    print('Raw')
    print(cm)
    print('Normalized')
    print(cm / np.sum(cm))
    plt.figure()
    plt.imshow(cm,  cmap=plt.cm.Blues)
    plt.colorbar()
    #tick_marks = np.arange(2)
    plt.xticks([-0.5,0.5], ['negative','positive'])
    plt.yticks([-0.5,0.5], ['negative','positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix\nPredicted vs Actual')
    return cm


def plot_auc(clf, X_train, y_train, X_test, y_test):
    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    i=0
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def get_metrics(R, label, show=True):
    avg = np.mean(R)
    st_dev = np.std(R)
    sharpe = avg / st_dev * 252 **.5
    metrics = ['mean','st_dev','Sharpe']
    values = [avg, st_dev, sharpe]
    pt = prettytable.PrettyTable(['metric','value'])
    for (m,v) in zip(metrics, values):
        pt.add_row( [m,v])
    if show:
        print('-' * len(label))
        print(label)
        print('-' * len(label))
        print(pt)
    return dict(zip(metrics,values))

def get_pnl_data(clf, records):
    X = np.array(list(map(lambda r: r.get(), records)))
    y = np.array(list(map(lambda r: r.get_fwd_vol(), records)))

    Y_hat = clf.predict(X)
    N = []
    pnl = []
    spx = []
    for n,(r,y_hat) in enumerate(zip(records, Y_hat)):
        ret = r.get_return()
        if y_hat > 0:
            pnl += [0]
        else:
            pnl += [ret]
        spx += [ret]
        N += [n]
    return (N, spx, pnl)

def get_pnl_plot(clf, r_in, r_out):
    (N_in, spx_in, pnl_in) = get_pnl_data(clf, r_in)
    (N_out, spx_out, pnl_out) = get_pnl_data(clf, r_out)
    
    get_metrics(pnl_in, 'Strategy (in sample)')
    get_metrics(spx_in, 'Buy And Hold (in sample)')
    
    get_metrics(pnl_out, 'Strategy (out of sample)')
    get_metrics(spx_out, 'Buy And Hold (out of sample)')
    
    N_out = N_in[-1]+  np.array([0] + N_out)
    spx_out = [sum(spx_in)] + spx_out
    pnl_out = [sum(pnl_in)] + pnl_out
    
    plt.plot(N_in, np.cumsum(pnl_in), label='strategy (in sample)', color='b')
    plt.plot(N_in, np.cumsum(spx_in), label='S&P (in sample)', color='k')
    
    plt.plot(N_out, np.cumsum(pnl_out), label='strategy (out of sample)', color='red')
    plt.plot(N_out, np.cumsum(spx_out), label='S&P (out of sample)', color='grey')
    
    plt.legend()
    plt.show()
    