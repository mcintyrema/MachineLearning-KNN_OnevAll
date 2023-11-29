import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import Reg_normalEqn as regNorm
import computeCost_ps4 as cc
import sklearn as skl
import sklearn.neighbors 


#loading in matlab file data
data1 = scipy.io.loadmat("input\hw4_data2.mat")
X1 = np.array(data1["X1"])
X2 = np.array(data1["X2"])
X3 = np.array(data1["X3"])
X4 = np.array(data1["X4"])
X5 = np.array(data1["X5"])
y1 = np.array(data1["y1"])
y2 = np.array(data1["y2"])
y3 = np.array(data1["y3"])
y4 = np.array(data1["y4"])
y5 = np.array(data1["y5"])

#first classifier
X_train1 = np.vstack((X1, X2, X3, X4))
y_train1 = np.vstack((y1, y2, y3, y4))
y_train1 = y_train1.reshape(-1)
X_test1 = np.array(X5)
y_test1 = np.array(y5)

#second classifier
X_train2 = np.vstack((X1, X2, X3, X5))
y_train2 = np.vstack((y1, y2, y3, y5))
y_train2 = y_train2.reshape(-1)
X_test2 = np.array(X4)
y_tes2 = np.array(y4)

#third classifier
X_train3 = np.vstack((X1, X2, X4, X5))
y_train3 = np.vstack((y1, y2, y4, y5))
y_train3 = y_train3.reshape(-1)
X_test3 = np.array(X3)
y_test3 = np.array(y3)

#fourth classifier
X_train4 = np.vstack((X1, X5, X3, X4))
y_train4 = np.vstack((y1, y5, y3, y4))
y_train4 = y_train4.reshape(-1)
X_test4 = np.array(X4)
y_test4 = np.array(y4)

#fifth classifier
X_train5 = np.vstack((X5, X2, X3, X4))
y_train5 = np.vstack((y5, y2, y3, y4))
y_train5 = y_train5.reshape(-1)
X_test5 = np.array(X1)
y_test5 = np.array(y1)

###Compute Average Accuracy###

#K=1:2:15
avgAccuracy = np.zeros(0)
for k in range(1, 16, 2):
    model = skl.neighbors.KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train1, y_train1)
    pred1 = model.predict(X_test1)
    #check differences between prediction and actual values
    accurate1 = np.sum(pred1==y1)

    # classifier 2
    model = skl.neighbors.KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train2, y_train2)
    pred2 = model.predict(X_test2)
    accurate2 = np.sum(pred2==y2)

    #classifier 3
    model = skl.neighbors.KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train3, y_train3)
    pred3 = model.predict(X_test3)
    accurate3 = np.sum(pred3==y3)

    # classifier 4
    model = skl.neighbors.KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train4, y_train4)
    pred4 = model.predict(X_test4)
    accurate4 = np.sum(pred4==y4)

    #classifier 5
    model = skl.neighbors.KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train5, y_train5)
    pred5 = model.predict(X_test5)
    accurate5 = np.sum(pred5==y5)

    avgAccuracy = np.append(avgAccuracy,(accurate1+accurate2+accurate3+accurate4+accurate5)/5)

k = np.arange(1, 16, 2)
pt.plot(k, avgAccuracy)
pt.show()