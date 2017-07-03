import sys
import os
os.chdir("C:/Users/behzad/Dropbox/_7_EduMaterial_PYTHON_MachineLearing/SVM/SVM-Classification-Terrain")

 
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl
from class_vis import prettyPicture, output_image



features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
#clf = SVC( C=1000,kernel="linear")
clf = SVC( C=100000.0,kernel="rbf")#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf.fit(features_train,labels_train)


#### store your predictions in a list named pred

pred = clf.predict(features_test)



from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
    
### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
