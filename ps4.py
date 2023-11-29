import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import Reg_normalEqn as regNorm
import computeCost_ps4 as cc

#loading in matlab file data
data1 = scipy.io.loadmat("input\hw4_data1.mat")

#500 features, 1000 samples
X = np.array(data1["X_data"])
#add bias feature
onesVector = np.ones((X.shape[0], 1))
# combine ones vector with score vectors to make feature matrix
X1 = np.hstack((onesVector, X))
y = np.array(data1["y"])
lambda1 = [0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017]
lambda1 = np.array(lambda1)
#print size of feature matrix X
print("Size of feature matrix X =", X1.shape)


trainingError = np.zeros((20, len(lambda1)))
testingError = np.zeros((20, len(lambda1)))
#compute average training and testing error from 20 different models trained on this data
for i in range(0,20):
    #grab 88% of x_train for training data
    size88 = int(0.88 * X1.shape[0])
    #getting random training sample for x1 by selecting indices at random w/o replacement
    X1_train_index = np.random.choice(np.arange(X1.shape[0]), size88, replace=False)
    X1_train = X1[X1_train_index]
    #getting random training sample for y
    y_train = y[X1_train_index]
    #grabbing other indices for X_test
    X_test_index = np.setdiff1d(np.arange(X1.shape[0]), X1_train_index)
    X_test = X1[X_test_index]
    #grabbing other idices for y
    y_test_index = np.setdiff1d(np.arange(y.size), X1_train_index)
    y_test = y[X_test_index]

    index = 0
    # training 8 linear regression models
    for j in lambda1:
        thetas = regNorm.Reg_normalEqn(X1_train, y_train, lambda1[index])
        #compute the training error
        trainingError[i][index] = cc.computeCost(X1_train, y_train, thetas)
        testingError[i][index] = cc.computeCost(X_test, y_test, thetas)
        index = index + 1

avgTrainingError = np.zeros(trainingError.shape[1])
avgTestingError = np.zeros(testingError.shape[1])
#compute average error of training and testing errors 
for k in range(0, trainingError.shape[1]):
    avgTrainingError[k] = np.mean(trainingError[:, k])
    avgTestingError[k] = np.mean(testingError[:, k])

print(avgTestingError)
print(avgTrainingError)
#plot training error vs lambda
pt.plot(lambda1, avgTrainingError, 'r-')
pt.plot(lambda1, avgTestingError, 'b-')
pt.show()