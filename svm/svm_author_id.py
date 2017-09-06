#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

from sklearn import svm
from sklearn.metrics import accuracy_score

##features_train = features_train[:len(features_train)/100]
##labels_train = labels_train[:len(labels_train)/100]

#clf = svm.SVC(kernel='linear')
c=10000.0
clf = svm.SVC(kernel='rbf',C=c)
print "C = ", c

# result with linear and 100% features
#	no. of Chris training emails: 7936
#	no. of Sara training emails: 7884
#	training time: 15.242 s
#	accuracy: 0.984072810011

# result with linear and 1% features
#	no. of Chris training emails: 7936
#	no. of Sara training emails: 7884
#	training time: 0.885 s
#	accuracy: 0.884527872582


t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "Predict time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(labels_test, pred)
print "accuracy:", accuracy



print "prediction for 10 = ", pred[10] # clf.predict(features_test[:10])
print "prediction for 26 = ", pred[26] # clf.predict(features_test[:26])
print "prediction for 50 = ", pred[50] # clf.predict(features_test[:50])


print "sara  (0) = ", (0==pred).sum()
print "chris (1) = ", (1==pred).sum()


#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#C =  10000.0
#training time: 99.986 s
#Predict time: 10.066 s
#accuracy: 0.990898748578
#prediction for 10 =  1
#prediction for 26 =  0
#prediction for 50 =  1
#sara  (0) =  881
#chris (1) =  877



