'''
A Python Script used for Data Science Research & Analysis in Kaggle competitions
'''

'''
Thoughts:
Customers are typically happy (As reflected with the high number of 0s).
Because customers are happy the unhappy customers may tend to have experienced some type of issue.
It is also possible customers with extreme asset amounts (very small or very large) are more likely to be critical.

Without much tuning it seems that the Linear Regression Model does Okay, and the XGboost model does the best so far.

Removing any number of columns has shown to reduce the accuracy of models, and adding new columns has shown to increase it.
Using reverse feature engineering and reading the problem description does show the some features to be highly correlated.

TODO: Anomoly detection
TODO: Sparsify memory objects - kg.X = kg.X.replace(0,np.nan).to_sparse()
'''

__author__ = 'Adam Tetelman'

import logging 
import numpy as np
import xgboost as xgb
from neat import nn, population, statistics, visualize

# TODO: If this is moved down below imports logging stops working. Seems to have to do with a broken  dependency in pybrain
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s') # DEBUG or INFO

from random import randint #Use to randomize a seed used
from pandas import DataFrame

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from sklearn import cross_validation
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import  VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize


'''Global Variables'''
NEW_FILE = False # Use newly created files; includes generated features
USE_TINY_FILES = False # Use small files; for testing scripting
LOW_MEMORY = False # For lower memory systems, some algorithms will run line by line rather than loading full data sets
SKIP_TEST = False # Run CV score only, do not score the Test data

'''which data/submission files to use for training and submissions'''
files = {}
files['submit'] = 'test_no_head.csv'
files ['mm-submit'] = 'mm.test_no_head.csv'
files['train'] = 'train_no_head.csv'
files['mm-train'] = 'mm.train_no_head.csv'

new_files = {}
new_files['train'] = 'train_no_head.csv'
new_files['submit'] = 'test_no_head.csv'
new_files ['mm-submit'] = 'mm.y_with_linear.csv'
new_files['mm-train'] = 'mm.x_with_linear.csv'

tiny_files = {}
tiny_files['submit'] = 'test_no_head.csv'
tiny_files ['mm-submit'] = 'mm.test_no_head.csv'
tiny_files['train'] = 'tiny_train_no_head.csv'
tiny_files['mm-train'] = 'mm.tiny_train_no_head.csv'

if USE_TINY_FILES:
    NEW_FILE = False

X_ROWS = 370 # ID, X1, X2, ..., ,VAL
if NEW_FILE:
    files = new_files
    X_ROWS = 738 # Includes new features; ID, X1, X2, ..., ,VAL

if USE_TINY_FILES:
    files = tiny_files


'''Start of KaggleData class'''
class KaggleData():
    
    def __init__(self):
        self.data = None
        self.data_submit = None
        self.Y_head = None
        
        self.models = {}
        
        self.X = None
        self.Y = None
        
        self.X_submit = None
        self.Y_submit = None
        
        self.X_train = None
        self.Y_train = None
        
        self.X_cv = None
        self.Y_cv = None
        
        self.X_test = None
        self.Y_test = None
        
    def read_data(self, mm_file, type, shape):
        '''Read train/test/cv data in from memmapped file'''
        logging.info('Reading data from %s.' %(mm_file))
        self.data = np.memmap(mm_file,dtype=type,mode='r',shape=shape)
        
    def read_submit_data(self ,mm_file, type, shape):
        logging.info('Reading submission data from %s.' %(mm_file))
        '''Read submition data in from memmapped file'''
        self.data_submit = np.memmap(mm_file,dtype=type,mode='r',shape=shape)
        
    def set_y_head(self, head):
        self.Y_head = head
        
    def set_xy_from_data(self):
        self.X = np.asarray(self.data[:,0:X_ROWS - 1]) # Array containing ID, X1...Xn
        self.Y = np.asarray(self.data[:,::X_ROWS]) # Array containing ID,y
            
    def set_xy_submit_from_data(self):
        self.Y_submit = None
        self.X_submit = np.asarray(self.data_submit[:,0:X_ROWS - 1]) # Array containing ID, X1...Xn   

    def split_data(self, cv, test, seed1, seed2):
        '''Split X into a training, cross validation, and test data set'''
        logging.info('Splitting data into train, cv, and test.')
        self.X_train, X_tmp, self.Y_train, Y_tmp = train_test_split(self.X,
            self.Y, test_size = (cv + test), random_state = seed1 )
        self.X_cv, self.X_test, self.Y_cv, self.Y_test = train_test_split(X_tmp,
            Y_tmp, test_size = (test/(cv + test)), random_state = seed2)
        logging.debug('X shape is %s. Y shape is %s.' %(self.X.shape, self.Y.shape))
        logging.debug('X_tmp shape is %s. Y_tmp shape is %s.' %(X_tmp.shape, Y_tmp.shape))
        logging.debug('X_train shape is %s. Y_train shape is %s.' %(self.X_train.shape, self.Y_train.shape))
        logging.debug('X_cv shape is %s. Y_cv shape is %s.' %(self.X_cv.shape, self.Y_cv.shape))
        logging.debug('X_test shape is %s. Y_test shape is %s.' %(self.X_test.shape, self.Y_test.shape))
        
    def reset_data(self):
        '''Reset all class data to unmoddified state from source file'''
        self.models = {}
        
        self.X = None
        self.Y = None

        self.X_submit = None
        self.Y_submit = None
        
        self.X_train = None
        self.Y_train = None
        
        self.X_cv = None
        self.Y_cv = None
        
        self.X_test = None
        self.Y_test = None

        self.set_xy_from_data()
        self.set_xy_submit_from_data()
                
    def clean_data(self, mode = 0):
        '''Clean data without doing much reduction. Different modes do things in different orders'''
        if mode == 0:
            self._variance_threshhold(0) # XXX: This may look ahead to test/cv data
            self._normalize()
        if mode == 1:
            self._normalize()
        if mode == 2:
            self._variance_threshhold(0) # XXX: This may look ahead to test/cv data
        if mode == 3:
            self._normalize()
            self._variance_threshhold(0) # XXX: This may look ahead to test/cv data
            
    def reduce_data(self, variance = .00001, features = 200, pca = 10, mode = 0):
        '''Destructive changes to the data to slim down 'unwanted' or 'overfitting' data points'''
        if mode == 0:
            self._remove_outliers()
            self._variance_threshhold(variance)
            self._select_features(features)
            self._pca(pca)
            
    def add_features(self, k = 2):
        '''Add various experimental new features to the data set'''
        # Add columns containing a k-means cluster label
        if k:
            self.add_kmeans_col(1000, 10, k)
            
        # Add columns containing X^2
        X_2 = self.X[:,1:] ** 2
        self.X = np.hstack((self.X, X_2))
        X_submit_2 = self.X_submit[:,1:] ** 2
        self.X_submit = np.hstack((self.X_submit, X_submit_2))        

        # Add columns containing 1/X
        X_2 = 1 / self.X[:,1:]
        self.X = np.hstack((self.X, X_2))
        X_submit_2 = 1 / self.X_submit[:,1:]
        self.X_submit = np.hstack((self.X_submit, X_submit_2))    
        
        # Scale down infinite values
        self.X_submit[self.X_submit >= 1E308] = 0
        self.X[self.X >= 1E308] = 0
        
    def write_submission(self, name):
        logging.info('Writing out submission file: ./results/%s' %(name))
        submission = DataFrame({"ID":[int(id) for id in self.Y_submit[:,0]], "TARGET":self.Y_submit[:,1]}) # XXX: This will only handle m x 1 Y
        submission.to_csv('results/' + name, index=False)
        
    def predict_y_submission(self, predictor, result=0, min_thresh=0, max_thresh=1,min=0, max=1):
        logging.info('Predicting Y(X) for submission.')
        y_submit = predictor(self.X_submit[:,1:]) # TODO: This won't support any Y larger than m x 1
        if result != 0:
            y_submit = y_submit[:,result] # XXX: If y contains probabilities for multiple categories take only 1                        
        y_submit = y_submit.ravel()
        y_submit[ y_submit < min_thresh] = min # Remove anything smaller than the min
        y_submit[ y_submit > max_thresh] = max # Remove anything larger than the max
        self.Y_submit = np.vstack((self.X_submit[:,0],y_submit[:])).T # Create an array ID,Y1,..,Yn]
    
    def _normalize(self):
        '''Normalize each column'''
        logging.info('Normalizing data.')
        self.X = np.hstack((self.X[:,0].reshape(-1,1),normalize(self.X[:,1:], norm='max', axis=0, copy=True)))
        self.X_submit = np.hstack((self.X_submit[:,0].reshape(-1,1),normalize(self.X_submit[:,1:], norm='max', axis=0, copy=True)))

    def _variance_threshhold(self, variance):
        '''Remove columns that do not meat the variance threshold'''
        logging.info('Removing data that has variance less than %f.' %(variance))
        vt = VarianceThreshold(variance)
        vt.fit(self.X) # XXX: Because idx should have high variance we pas all of X
        self.X = vt.transform(self.X)
        self.X_submit = vt.transform(self.X_submit)
        
        # Repeat this process for X_submit # XXX: This might not be kosher outside of competition
        vt.fit(self.X_submit)
        self.X = vt.transform(self.X)
        self.X_submit = vt.transform(self.X_submit)
        
    def _remove_outliers(self):
        pass # TODO:
    
    def _pca(self, n):
        '''Reduce X and X_submit down to n principle components'''
        logging.info('Reducing X down to %d principle components.' %(n))
        if n >= self.X.shape[1]:
            logging.warn('Number of features is greater than/equal to  n.')
        else:
            pca = PCA(n_components=n)
            pca.fit(self.X[:,1:])
            self.X = np.column_stack(  (self.X[:,0], pca.transform(self.X[:,1:])))
            self.X_submit = np.column_stack( (self.X_submit[:,0], pca.transform(self.X_submit[:,1:])) )
            
    def _select_features(self, n):
        '''Reduce X to the n best features that represent Y'''
        logging.info('Reducing X from %d features to %d.' %(self.X.shape[1],n))
        if n >= self.X.shape[1]:
            logging.warn('Number of features is greater than/equal to  n.')
        else:
            sk = SelectKBest(k=n)
            sk.fit_transform(self.X[:,1:],self.Y[:,1]) # XXX: This will look ahead to cv/test data
            sk.transform(self.X_submit[:,1:])
            
    def __score_cv(self, model, predict, result = 0):
        logging.info('Scoring CV.')
        if True:                
            if result != 0: # XXX: This only works for 1 Y result
                s = roc_auc_score(self.Y_cv[:,1:], predict(self.X_cv[:,1:])[:,result])
            else:
                b = predict(self.X_cv[:,1:])
                s = roc_auc_score(self.Y_cv[:,1:], b )
        logging.info('Cross-Validation Scored: %f.' %(s))
        return s
    
    def __score_test(self, model, predict, result = 0, xg = 0):
        logging.info('Scoring test.')
        if SKIP_TEST:
            s = None
        else:
            if True: 
                if result != 0: # XXX: This only works for 1 Y result
                    s = roc_auc_score(self.Y_test[:,1:], predict(self.X_test[:,1:])[:,result])
                else:
                    s = roc_auc_score(self.Y_test[:,1:], predict(self.X_test[:,1:]))
            logging.info('Test Scored: %f.' %(s))    
        return s    
        
    def __score_test_mean(self, model, predict, result = 0):
        logging.info('Scoring test.')
        low_memory = LOW_MEMORY
        if low_memory: # In low memory systems we use the slower method of iterating over rows
            count = 0
            sum = 0
            for i in range(0,self.X_test.shape[0]):
                count = count + 1
                if result != 0: # XXX: This only works for 1 Y result
                    sum = sum + (predict(self.X_test[i,1:].reshape(1,-1))[result] - self.Y_test[i,1:][result]) ** 2
                else:
                    sum = sum + (predict(self.X_test[i,1:].reshape(1,-1)) - self.Y_test[i,1:]) ** 2
            s = 1 - sum / count
        else: 
            if result != 0:
                s = 1 - np.mean((predict(self.X_test[:,1:])[:,result] - self.Y_test[:,1:]) ** 2)
            else:
                s = 1 - np.mean((predict(self.X_test[:,1:]) - self.Y_test[:,1:]) ** 2)
        logging.info('Test Scored: %f.' %(s))
        
    def score(self,h,y):
        '''Report the Area Under Curve Score'''
        s = roc_auc_score(y,h)
        logging.info('Score of %f' %(s))

    def score_mean(self, h, y):
        '''Report the Absoulte Means Error Score'''
        low_memory = LOW_MEMORY
        if low_memory:
            count = 0
            sum = 0
            for i in range(0,h.shape[0]):
                sum = sum + (h[i] - y[i]) ** 2
                count = count + 1
            s = 1 - sum / count
        else:
            s = 1 - np.mean((h[i] - y[i]) ** 2)
        logging.info('Score of %f' %(s))
            
    def __fit(self, model, fit):
        logging.info('Fitting Model.')
        fit(self.X_train[:,1:], self.Y_train[:,1:].reshape(-1,1)) # TODO: This will only work for y of len 1, but it avoids a dataconversionwarning

    def random_forest_classifier(self, n, s, j=-1, c='gini', w='balanced'):
        logging.info('Beginning RFC model.')
        rfc = RandomForestClassifier(n_estimators=n, random_state=s, n_jobs=j,criterion=c, class_weight=w)
        self.__fit(rfc,rfc.fit) # INFO: This should do this rfc.fit(self.X_train[:,1:], selt.Y_train[:,1:])
        y = self.Y_cv[:,1:]
        self.__score_cv(rfc,rfc.predict_proba,1)        
        self.__score_test(rfc,rfc.predict_proba,1)
        self.predict_y_submission(rfc.predict_proba,1)
        self.write_submission('rfc.csv')
        self.models['rfc'] = rfc
        logging.info('Completed RFC model.')
        return rfc
    
    def add_linear_regression(self):
        lr = self.linear_regression()
        lr = linear_model.LogisticRegression()
        lr.fit(normalize(self.X_train[:,1:]), self.Y_train[:,1])
        self.X_train = np.hstack((self.X_train, lr.predict(normalize(self.X_train[:,1:])).reshape(-1,1))) 
        self.X_cv = np.hstack((self.X_cv, lr.predict(normalize(self.X_cv[:,1:])).reshape(-1,1))) 
        self.X_test = np.hstack((self.X_test, lr.predict(normalize(self.X_test[:,1:])).reshape(-1,1))) 
        self.X_submit = np.hstack((self.X_submit, lr.predict(normalize(self.X_submit[:,1:])).reshape(-1,1))) 

    def linear_regression_2(self):
        '''Run a linear regression and save the output of that regression as new X features'''
        logging.info('Beginning Creation of new files based on Linear Regression model.')
        x_2 = self.X[:,1:] ** 2
        y_2 = self.X_submit[:,1:] ** 2
        
        self.reset_data()
        self.clean_data()
        kg.split_data(.3, .0001, 100, 100)
        lr = linear_model.LinearRegression()        
        self.__fit(lr,lr.fit) 
        self.__score_cv(lr,lr.predict)        
        x_pred = lr.predict(self.X[:,1:])
        y_pred = lr.predict(self.X_submit[:,1:])
        self.reset_data()
        
        X_new = np.hstack((self.X,x_pred))
        Y_new = np.hstack((self.X_submit, y_pred))
        
        X_new = np.hstack((X_new,x_2))
        Y_new = np.hstack((Y_new, y_2))

        X_new = np.hstack((X_new,self.Y[:,1].reshape(-1,1)))
        
        logging.info('New X shape is %s' %(str(X_new.shape)))
        logging.info('New Y shape is %s' %(str(Y_new.shape)))
        mm = np.memmap('mm.x_with_linear.csv', dtype='float32', mode='w+',shape=X_new.shape)
        mm[:] = X_new[:]
        del mm
        mm = np.memmap('mm.y_with_linear.csv', dtype='float32', mode='w+',shape=Y_new.shape)
        mm[:] = Y_new[:]
        del mm
        logging.info('Completed creating new files based on Linear Regression model. Remember to update X_ROW values.')
    
    def linear_regression(self):
        logging.info('Beginning Linear Regression model.')
        lr = linear_model.LinearRegression(fit_intercept = True)
        #lr = linear_model.LinearRegression() 
        self.__fit(lr,lr.fit) 
        self.__score_cv(lr,lr.predict)        
        self.__score_test(lr,lr.predict)
        self.predict_y_submission(lr.predict)
        self.write_submission('linreg.csv')
        self.models['linreg'] = lr
        logging.info('Completed Linear Regression model.')
        return lr
        
    def logistic_regression(self):
        logging.info('Beginning Logistic Regression model.')
        lr = linear_model.LogisticRegression(solver = 'lbfgs', fit_intercept = False,  C = .95, dual = False, penalty = 'l2', multi_class = 'multinomial')
        #lr = linear_model.LogisticRegression()
        self.__fit(lr,lr.fit) 
        self.__score_cv(lr,lr.predict_proba,1)        
        self.__score_test(lr,lr.predict_proba,1)
        self.predict_y_submission(lr.predict_proba,1)
        self.write_submission('logreg.csv')
        self.models['logreg'] = lr
        logging.info('Completed Logistic Regression model.')
        return lr
    
    def kmeans_linear_regression(self, iter = 1000, n_init = 10, n = 4, model = 0):
        logging.info('Starting kmeans+  model.')
        self.kmeans_clustering(iter,n_init,n,model)
        logging.info('Scoring %d kmeans+ regression model.' %(n))        
        self.__score_cv(self.models['km'], self.kmeans_predict, self.km_result)
        self.__score_test(self.models['km'], self.kmeans_predict, self.km_result)
        self.predict_y_submission(self.kmeans_predict, self.km_result)
        self.write_submission('kmeans-' + str(n) + '.csv')
        logging.info('Completed kmeans+ regression model.')        
        
    def kmeans_investigate(self, iter=100, n_init = 10, n = 10):
        '''Used to manually investigate k-means clustering, prints member count of each cluster'''
        for n in range(9,n):
            km = KMeans(n_clusters=n, max_iter=iter, n_init=n_init)
            km.fit(self.X_train[:,1:])
            X_cluster = km.predict(self.X_train[:,1:])
            count = {}
            for i in range(0,n):
                count[i]=0
            for i in X_cluster:
                count[X_cluster[i]] = 1 + count[X_cluster[i]]
            logging.info('cluster count %d:' %(n))  
            for i in range(0,n):                
                if count[i] != 0:
                    logging.info("%d:%d"%(i,count[i]))
                    
    def add_kmeans_col(self, iter = 1000, n_init = 10, n = 4):
        '''Add a new k_means cluster column to X data'''
        logging.info('Adding kmeans %d clusters to X' %(n))
        km = KMeans(n_clusters=n, max_iter=iter, n_init=n_init)
        km.fit(self.X[:,1:]) # XXX: This might not be kosher as it affects all of X
        self.models['km-col'] = km        
        self.X = np.hstack( (self.X, km.predict(self.X[:,1:]).reshape(-1,1)) )   
        
    def kmeans_clustering(self, iter=1000, n_init = 10, n = 4, model = 0, test_type = 3):
        '''Run some kmeans clustering followed by training a different model on each cluster'''
        km = KMeans(n_clusters=n, max_iter=iter, n_init=n_init)
        km.fit(self.X_train[:,1:])
        self.models['km'] = km
        
        X_train_tmp = self.X_train 
        Y_train_tmp = self.Y_train
        if model == 0:
            lr_default = self.linear_regression()
        if model == 1:
            lr_default = self.random_forest_classifier(100,44)
        if model == 2:
            lr_default = self.logistic_regression()

        X_cluster = km.predict(self.X_train[:,1:])
        count = {}
        
        for i in range(0,n):
            count[i]=0
        for i in X_cluster:
            count[X_cluster[i]] = 1 + count[X_cluster[i]]
        for i in range(0,n):
            if count[i] < 6000 and count[i] > 1000:
                self.km_small = i
                logging.debug('Found Small cluster %d of size %d' %(i, count[i]))
        for i in range(0,n):
            if count[i] > 10000:
                self.km_big = i
                break
        self.models['km-models'] = {}
        
        if model == 0:
            self.km_result = 0 #1 for rfc/logreg, 0 for linreg
        else:
            self.km_result = 1 #1 for rfc/logreg, 0 for linreg
            
        if test_type == 1: # TODO: Collapse this if/else into a single function
            self.X_train = []
            self.Y_train = []
            for i in range(0,len(X_train_tmp)):
                cluster = km.predict(X_train_tmp[i,1:].reshape(1,-1))[0]
                if cluster != self.km_small:
                    self.X_train.append(X_train_tmp[i,:])
                    self.Y_train.append(Y_train_tmp[i,:])
            self.X_train = np.asarray(self.X_train)
            self.Y_train = np.asarray(self.Y_train)

            logging.debug('Running model on cluster %d with below X and Y shapes.' %(i))
            logging.debug(self.X_train.shape)
            logging.debug(self.Y_train.shape)
            if model == 0:
                lr = self.linear_regression()
            if model == 1:
                lr = self.random_forest_classifier(100,44)
            if model == 2:
                lr = self.logistic_regression()
            for i in range(0,n):
                self.models['km-models'][i] = lr
            self.X_train = X_train_tmp
            self.Y_train = Y_train_tmp

            self.X_train = [] ### XXX: self.X_train = self.X_train[km.predict(self.X_train[:,1:]) == self.km_small]
            self.Y_train = [] ### XXX: self.Y_train = self.Y_train[km.predict(self.X_train[:,1:]) == self.km_small]
            
            for j in range(0,len(X_train_tmp)):
                cluster = km.predict(X_train_tmp[j,1:].reshape(1,-1))[0]
                if cluster == self.km_small:
                    self.X_train.append(X_train_tmp[j,:])
                    self.Y_train.append(Y_train_tmp[j,:])
            self.X_train = np.asarray(self.X_train)
            self.Y_train = np.asarray(self.Y_train)
            logging.debug('Running model on cluster %d with below X and Y shapes.' %(i))
            logging.debug(self.X_train.shape)
            logging.debug(self.Y_train.shape)
            if model == 0:
                lr = self.linear_regression()
            if model == 1:
                lr = self.random_forest_classifier(100,44)
            if model == 2:
                lr = self.logistic_regression()
            self.models['km-models'][self.km_small] = lr
            self.X_train = X_train_tmp
            self.Y_train = Y_train_tmp
            return 
    
        for i in range(0,n):
            logging.info('Constructing/training model %d on cluster %d' %(model, i))
            self.X_train = [] ### XXX: self.X_train = self.X_train[km.predict(self.X_train[:,1:]) == i]
            self.Y_train = [] ### XXX: self.Y_train = self.Y_train[km.predict(self.X_train[:,1:]) == i]
            
            count = 0
            for j in range(0,len(X_train_tmp)):
                cluster = km.predict(X_train_tmp[j,1:].reshape(1,-1))[0]
                if cluster == i:
                    count = count + 1
                    self.X_train.append(X_train_tmp[j,:])
                    self.Y_train.append(Y_train_tmp[j,:])
            self.X_train = np.asarray(self.X_train)
            self.Y_train = np.asarray(self.Y_train)
            
            if count == 0:
                self.models['km-models'][i] = lr_default
                continue
            logging.debug('Running model %d on cluster %d with below X and Y shapes.' %(model, i))
            logging.debug(len(self.X_train))
            logging.debug(len(self.Y_train))
            if model == 0:
                lr = self.linear_regression()
            if model == 1:
                lr = self.random_forest_classifier(100,44)
            if model == 2:
                lr = self.logistic_regression()
            if model == 3:
                lr = self.xgboost()
            self.models['km-models'][i] = lr
            self.X_train = X_train_tmp
            self.Y_train = Y_train_tmp

    def kmeans_predict(self, x):
        y = []
        for i in range(0,x.shape[0]):
            cluster = self.models['km'].predict(x[i,:].reshape(1,-1))[0]
            if self.km_result != 0: # TODO: Clean this up a bunch
                y.append(self.models['km-models'][cluster].predict_proba(x[i,:].reshape(1,-1))[0])
            else:
                y.append(self.models['km-models'][cluster].predict(x[i,:].reshape(1,-1))[0][0])
            # TRACE #logging.debug('Value %d belongs to cluster %d with predicted value %f' %(i, cluster, y[-1:][0]))
        y = np.asarray(y)
        return y

    def xgboost(self, loop = False):
        logging.info('Beginning of XG Boost model.')
        s = []
        for i in [.02,.03,.035,.04,.05]: # .035 and .045 seemed to work well
            for j in [3,4,8,10,20]: #4 (maybe 10) seemed to be the peak
                for k in [.6,.7,.8,.95]: #.8 or .95
                    for e in [.4,.6,.7,.75]: # 75 seemed best .85 seemed okay 
                        for n in [1000]: # Higher seems to work better, loss of returns seems around 650
                            seed = randint(0,10000)
                            xgbc = xgb.XGBClassifier(silent=True,missing=np.nan, max_depth=j, n_estimators=n, learning_rate=i, nthread=8, subsample=k, colsample_bytree=e, seed=seed)
                            logging.info('Fitting Model.') # self.__fit(xgbc,xgbc.fit)
                            xgbc.fit(self.X_train[:,1:],  self.Y_train[:,1:], early_stopping_rounds=20, eval_metric="auc", eval_set=[(self.X_cv[:,1:], self.Y_cv[:,1:])])
                            b = self.__score_cv(xgbc,xgbc.predict_proba,1)        
                            self.models['xgb'] = xgbc
                            if loop:
                                logging.debug({'i':i,'j':j,'k':k,'e':e,'n':n,'seed':seed,'score':b})
                            else:
                                b_2 = self.__score_test(xgbc,xgbc.predict_proba,1)                                
                                self.predict_y_submission(xgbc.predict_proba,1)
                                self.write_submission('xgb.csv')
                            logging.info('Completed XG Boost model.')
                            return b
                        
    def neat(self, n = 600):
        class ThisNeat():
            # http://neat-python.readthedocs.org/en/latest/config_file.html
            # http://neat-python.readthedocs.org/en/latest/xor_example.html#running-neat
            def __init__(self, n, kg):
                self.pop = population.Population('kg-neat.config') # XXX: External config file
                self.n = n
                self.winner = None
                self.winner_net = None
                self.kg = kg
                
            def eval_fitness(self,genomes):
                for g in genomes:
                    net = nn.create_feed_forward_phenotype(g)
                    output = []
                    for x in self.fit_X:
                        l = net.serial_activate(x)
                        count = 0
                        sum = 0
                        for j in l:
                            #print(l)
                            sum += j
                            count += 1
                        k = sum/count
                        output.append(k)
                    error = roc_auc_score(self.fit_Y,output)
                    g.fitness = 1 - error
                    
            def fit(self, X, Y):
                self.fit_X = X # XXX: store X_Train as a tmp class var
                self.fit_Y = Y # XXX: store Y_Train as a tmp class var
                self.pop.run(self.eval_fitness,self.n)
                self.winner = self.pop.statistics.best_genome()
                self.winner_net = nn.create_feed_forward_phenotype(self.winner)

            def predict(self, X):
                output = []
                for x in X:
                    l = self.winner_net.serial_activate(x)
                    sum = 0
                    count = 0
                    for j in l:
                        sum += j
                        count += 1
                    k = sum/count
                    output.append(k)
                return np.asarray(output)
            
        logging.info('Starting NEAT model.')
        kgneat = ThisNeat(n,self)
        self.__fit(kgneat,kgneat.fit) 
        self.__score_cv(kgneat,kgneat.predict)        
        self.__score_test(kgneat,kgneat.predict)
        self.predict_y_submission(kgneat.predict)
        self.write_submission('neat.csv')
        self.models['neat'] = kgneat
        logging.info('Completed NEAT model.')
        return kgneat
        
    def nn_1(self):
        logging.info('Beginning Neural Network model.')
        
        class ThisNN(): # Used to abstract away fit function
            def __init__(self, nn, kg):
                self.nn = nn
                self.kg = kg
                
            def fit(self, X, Y):
                logging.info('Generating a Pybrain SupervisedDataSet')
                ds = SupervisedDataSet(X,Y)
                trainer = BackpropTrainer(self.nn,ds)
                for i in range(0,10):
                    logging.debug(trainer.train()) # XXX Runs once
                logging.info('Training Neural Network until Convergence')

                cv = SupervisedDataSet(self.kg.X_cv[:,1:],self.kg.Y_cv[:,1:])
                trainer.trainUntilConvergence(verbose=11, validationData=cv, trainingData=ds)
            
            def predict_x(self, X):
                Y = []
                for i in range(0,X.shape[0]):
                    Y.append(self.nn.activate(X[i,:]))
                return np.asarray(Y)

        net = buildNetwork(self.X_train.shape[1] - 1,3,1) # X - 1 to avoid ID
        this_nn = ThisNN(net,self) 
        self.__fit(net,this_nn.fit) 
        self.__score_cv(net,this_nn.predict_x)        
        self.__score_test(net,this_nn.predict_x)
        self.predict_y_submission(this_nn.predict_x)
        self.write_submission('nn.csv')
        self.models['nn'] = net
        logging.info('Completed Neural Network model.')
        return net


def score_zeros(kg):
    '''baseline score of all zeros'''
    h = np.copy(kg.Y)
    h[:,1] = 0
    kg.score(h[:,1],kg.Y[:,1])


def do_xg(kg):
    '''Exploratory function to find the best cleaning/reduction/run methods using xgboost'''
    s = []
    max = 0
    while True:
        for clean in [0,1,2,3,4]: # Not doing any normalizing and retaining all features give best results
            for reduce in [False]:
                for pca in [400,200,100,50,30,15]:
                    for features in [400,300,250,200,150,100,50,30,15]:
                        for k in [2]: # 2 clusters seems to give a decent boost
                            for var in [.0001,.001]:
                                for l in [.2]:
                                    kg.reset_data()
                                    kg.clean_data(clean)
                                    if reduce:
                                        kg.reduce_data(var, features, pca)
                                    if k:
                                        kg.add_features(k)
                                    kg.split_data(l, .0005, 100, 100)
                                    
                                    if False:
                                        kg.add_linear_regression()
                                    try:
                                        score = kg.xgboost()
                                        l = {'xg': score, 'clean':clean,'pca':pca,'features':features,'k':k,'var':var}
                                    except Error:
                                        score = 0
                                        l = {}
                                        
                                    s.append(l)
                                    logging.debug(s)
                                    if score > max:
                                        max = score
                                        kg.predict_y_submission(kg.models['xgb'].predict_proba,1)
                                        kg.write_submission(str(score) + '-xgb.csv')
        for i in s:
            logging.debug(i)

    
kg = KaggleData()

# To speed up loading data from disk between runs the files have been memmammped
kg.read_data(files['mm-train'], 'float32', (76020,X_ROWS + 1))
kg.read_submit_data(files['mm-submit'], 'float32', (75818,X_ROWS))

kg.reset_data() #Set class variables from read in submition/training data
kg.clean_data()
kg.reduce_data(.00001, 400, 400) # Does not seem helfpule
kg.add_kmeans_col(100,10,5) # Not necessary when using NEW_FILE
kg.split_data(.2, .2, 100, 100)

# Run a bunch of algorithms
score_zeros(kg)
kg.logistic_regression()
kg.kmeans_investigate()
kg.linear_regression()
kg.random_forest_classifier(300,10)
kg.kmeans_linear_regression(100,10,2,1)
kg.neat()
kg.nn_1()
kg.xgboost()
kg.neat()
do_xg(kg)
exit()