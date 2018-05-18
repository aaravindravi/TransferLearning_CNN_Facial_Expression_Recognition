# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:57:42 2018
Finalised
@author: Aravind
Final Submission for VGG19 - Using the features of Block3 Pooling Layer
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
import pickle

#Loading the Features
X1 = np.load('class1vgg19_data.npy')
X2 = np.load('class2vgg19_data.npy')
X3 = np.load('class3vgg19_data.npy')
X4 = np.load('class4vgg19_data.npy')
X5 = np.load('class5vgg19_data.npy')
X6 = np.load('class6vgg19_data.npy')
X7 = np.load('class7vgg19_data.npy')

#Concatenate into single matrix
X_data = np.append(X1,X2,axis=0)
X_data = np.append(X_data,X3,axis=0)
X_data = np.append(X_data,X4,axis=0)
X_data = np.append(X_data,X5,axis=0)
X_data = np.append(X_data,X6,axis=0)
X_data = np.append(X_data,X7,axis=0)

#Labels
Y_data = X_data[:,(X_data.shape[1]-1)]

#Traning Data
X_data = X_data[:,0:(X_data.shape[1]-1)]

#Unseen Test Data
X_test = np.load('class8vgg19_data.npy');
X_test = X_test[:,0:(X_test.shape[1]-1)]

#Dimensionality Reduction on Features
pca = PCA(n_components=200)
components_pca = pca.fit(X_data)
X_data1= pca.transform(X_data)
X_test1= pca.transform(X_test)

#Train Test Split 80-20
x_tr,x_ts,y_tr,y_ts = train_test_split(X_data1, Y_data, test_size=0.2,random_state=83)
#87 6 83 79 26 18

#Classifier SVM Linear Kernel 
clf = LinearSVC(C=10)
clf = clf.fit(x_tr,y_tr)
predictions_tr = (clf.predict(x_ts))

#10-Fold Crossvalidation Accuracy
scores = cross_val_score(clf, x_tr, y_tr, cv=10)
print("Training Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''
#Leave One Out or Jack-Knife Crossvalidation
loo_train_acc=[]
loo = LeaveOneOut()
for train_index, test_index in loo.split(x_tr):
   X_train, X_test = x_tr[train_index], x_tr[test_index]
   y_train, y_test = y_tr[train_index], y_tr[test_index]
   clf = clf.fit(X_train,y_train)
   predictions = (clf.predict(X_test))
   loo_train_acc.append(accuracy_score(y_test,predictions))

loo_train_accuracy = np.asarray(loo_train_acc)
print("LOO Accuracy: %0.4f" % loo_train_accuracy.mean())
'''
#20% Test Data Accuracy
test_acc = accuracy_score(y_ts,predictions_tr)
print("Test Accuracy: %0.4f" % test_acc)

#Predictions on Unseen Data
predictions_new = (clf.predict(X_test1))

#Save the PCA parameters and SVM Model
pickle.dump(clf, open('FER_vgg19net_ck_transfer_only_svm.sav', 'wb'))
pickle.dump(pca, open( "pca_ck.p", "wb" ) )
