from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Global Vars
debug = False

class KaggleData():
    
    def __init__(self,csv,delim):
        print('Loading in file: ' + csv + ', delimited by: ' + delim)
        self.data = np.loadtxt(csv,delimiter=delim)
        self.data_mz = np.ma.masked_equal(self.data, 0)

        print('Data loaded.')
    
    def hist(self, name="", mask=None):
        print("Generating Feature Histogram Plots.")
        for c in range(0, self.data.shape[1]):
            mean = self.data[:,c].mean()
            mode = stats.mode(self.data[:,c])
            plt.hist(np.ma.masked_equal(self.data[:,c], mask))
            plt.axvline(mean,color='b',linestyle='dashed', linewidth=2)
            plt.axvline(mode[0],color='r',linestyle='dashed', linewidth=2)
            plt.savefig(name + 'hist-feature-' + str(c) + ".png")
            plt.clf()
            print(c)
        print("Histograms Generated.")
        
    def box(self, name="", mask=None):
        print ("Generating Box Plots.")
        for c in range(0, self.data.shape[1]):
            plt.boxplot(np.ma.masked_equal(self.data[:,c], mask))
            plt.savefig(name + 'box-feature-' + str(c) + ".png")
            plt.clf()
        print("Boxes Generated.")
        
    def heat(self, name="", mask=None):
        print("Generating Heat Map.")
        plt.pcolor(np.ma.masked_equal(self.data, mask), cmap=plt.cm.Reds)
        plt.savefig(name + 'heat-feature-all.png')
        plt.clf()
        for c in range(0, self.data.shape[1]):
            plt.pcolor(np.ma.masked_equal(self.data[:,c], mask), cmap=plt.cm.Reds)
            plt.savefig(name + 'heat-feature-' + str(c) + ".png")
            plt.clf()
        plt.clf()
        print("Heat Map Generated.")
        
        
def test():
    input = 'train_no_head.csv'
    if debug:
        input = 'tiny_train_no_head.csv'
    kd = KaggleData(input, ',')
    
    def do_graphs(name="",mask=None):
        #Create graphs
        kd.hist(name,mask)
        kd.box(name,mask)
        kd.heat(name) # This takes up a lot of memory

    do_graphs("images/")
    do_graphs("images/mz", 0)
    
    kd = KaggleData('test_no_head.csv', ',')
    do_graphs("images/test")
    do_graphs("images/testmz", 0)

if __name__ == '__main__':
    test()