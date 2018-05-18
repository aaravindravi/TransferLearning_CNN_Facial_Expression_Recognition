# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:57:42 2018
Finalised
@author: Aravind
Final Submission for VGG19 - Using the features of Block3 Pooling Layer CK
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import pickle
pca_components = [50,100,150,200]
layers_to_extract = ["block1_pool","block2_pool","block3_pool","block4_pool","block5_pool","fc1"]

for pca_num in range(0,4):
    print("PCA Components: %0.1f" % pca_components[pca_num])
    for layer_num in range(0,6):
    
        #Loading the Features
        X1 = np.load("class_jaffe"+layers_to_extract[layer_num]+"1vgg19_data.npy")
        X2 = np.load("class_jaffe"+layers_to_extract[layer_num]+"2vgg19_data.npy")
        X3 = np.load("class_jaffe"+layers_to_extract[layer_num]+"3vgg19_data.npy")
        X4 = np.load("class_jaffe"+layers_to_extract[layer_num]+"4vgg19_data.npy")
        X5 = np.load("class_jaffe"+layers_to_extract[layer_num]+"5vgg19_data.npy")
        X6 = np.load("class_jaffe"+layers_to_extract[layer_num]+"6vgg19_data.npy")
        X7 = np.load("class_jaffe"+layers_to_extract[layer_num]+"7vgg19_data.npy")
        
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
        X_test = np.load("class_jaffe"+layers_to_extract[layer_num]+"8vgg19_data.npy");
        X_test = X_test[:,0:(X_test.shape[1]-1)]
        
        #Dimensionality Reduction and Feature Selection on Layer Features
        pca = PCA(n_components=pca_components[pca_num])
        components_pca = pca.fit(X_data)
        X_data1= pca.transform(X_data)
        X_test1= pca.transform(X_test)
        
        #Train Test Split 80-20
        x_tr,x_ts,y_tr,y_ts = train_test_split(X_data1, Y_data, test_size=0.2,random_state=124)
         #87 6 83 79 26 18 ## 79 #124
        
        #Classifier SVM Linear Kernel 
        clf = LinearSVC(C=10)
        clf = clf.fit(x_tr,y_tr)
        predictions_tr = (clf.predict(x_ts))
        
        #10-Fold Crossvalidation Accuracy
        scores = cross_val_score(clf, x_tr, y_tr, cv=10)
        print(layers_to_extract[layer_num])
        print("Training Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
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
        
        #20% Test Data Accuracy
        test_acc = accuracy_score(y_ts,predictions_tr)
        print("Test Accuracy: %0.4f" % test_acc)
        
        #Predictions on Unseen Data
        #predictions_new = (clf.predict(X_test1))
        
        #Save the PCA parameters and SVM Model
        #pickle.dump(clf, open('FER_vgg19net_ck_transfer_only_svm.sav', 'wb'))
        #pickle.dump(pca, open( "pca_ck.p", "wb" ) )
