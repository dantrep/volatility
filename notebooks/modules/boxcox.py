'''
Manage Data Splitting and Box Cox Transforms

X = raw time series
self.Y = transformed time series

'''
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from prettytable import PrettyTable
import matplotlib.pyplot as plt

class BoxCox(object):
    def __init__(self, X, test_size):
        assert min(X) > 0, 'only positive values can be transformed'
        self.X = X
        self.train, self.test = train_test_split(X, test_size=test_size, shuffle=False)
        try:
            self.Y, self.lbda = stats.boxcox(self.train)
        except:
            print('WARNING: unable to transform the data')
            self.Y, self.lbda = (self.train, None)
        print('Sampling       :  %d / %d\nBox Cox lambda : %s' % (len(self.Y), len(X), self.lbda))
    
    def plot(self):
        fig = plt.figure()
        ax_1 = fig.add_subplot(211)
        #X = air['Passengers'].tolist()
        prob = stats.probplot(self.train, dist=stats.norm, plot=ax_1)
        ax_1.set_title('Probability vs Normal\nRaw Data')
        ax_1.axes.get_xaxis().set_visible(False)
        
        ax_2 = fig.add_subplot(212)
        prob = stats.probplot(self.Y, dist=stats.norm, plot=ax_2)
        ax_2.set_title('After Box-Cox transformation')
        plt.show()
        
        header = ['data','mean','st_dev','std to mean']
        pt = PrettyTable(header)
        pt.add_row(['Raw', np.mean(self.train), np.std(self.train), np.std(self.train) / np.mean(self.train)])
        if self.lbda is not None:
            print('\nBox Cox Transform lambda : %f'% self.lbda)
            pt.add_row(['Box Cox', np.mean(self.Y), np.std(self.Y), np.std(self.Y) / np.mean(self.Y)])
        else:
            print('no tranformation!')
        print(pt)
    
    def get_test(self):
        return list(map(lambda x: self.apply(x), self.test))
    
    def apply(self, x):
        if self.lbda is None:
            return x
        elif self.lbda == 0:
            return np.log(x)
        else:
            return (x ** self.lbda - 1) / self.lbda
    
    def unapply(self, y):
        if self.lbda is None:
            return y
        elif self.lbda == 0:
            return np.exp(y)
        else:
            return (self.lbda * y + 1) ** (1/self.lbda)
