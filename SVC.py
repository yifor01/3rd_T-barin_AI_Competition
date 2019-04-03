# -*- coding: utf-8 -*-
import winsound
from sklearn import svm
from sklearn.model_selection import GridSearchCV

def svc_Class_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, 
                               scoring='accuracy', verbose = 2,n_jobs=2)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


svc_Class_param_selection(class_data, class_y, 5 )

 import os
 os.chdir('D:\google下載\libsvm-3.23\libsvm-3.23\python')
 from svmutil import *
 
 
 
 y,x = svm_read_problem('../heart_scale')
 m = svm_train(y[:200],x[:200],'-c 4')
 p_label,p_acc,p_val=svm_predict(y[200:],x[200:],m)



[c,g]=[2^-10,2^10]*[2^-10,2^10]

svm_train( class_y.values,class_data,'-c 2' )

svm_predict(class_pred)
