import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.linear_model import LogisticRegression
import tabulate

#loading in matlab file data
data3 = scipy.io.loadmat("input\hw4_data3.mat")
# print(data3.keys())

X = np.array(data3["X_train"])
#add bias feature
onesVector = np.ones((X.shape[0], 1))
# combine ones vector with x matrix to make feature matrix
X1 = np.hstack((onesVector, X))
y = np.array(data3["y_train"])
xtest = np.array(data3["X_test"])
onesVector = np.ones((xtest.shape[0], 1))
xtest1 = np.hstack((onesVector, xtest))
y_test = np.array(data3["y_test"])
print(X1.shape)
def logReg_multi(X_train, y_train, X_test):
    #detect different classes
    classes = []
    for i in y_train:
        if i[0] not in classes:
            classes.append(i[0])
    classes = np.array(classes)
    
    trained = []
    #make classes 1 or 0
    for j in classes:
        label = np.where(y_train == j, 1, 0) #samples of indexed class set to 1
        #training if sample belongs to indexed class
        mdl = LogisticRegression(random_state=0).fit(X_train, label.ravel())
        trained.append(mdl) #append three classes that were identified in y_train

    #initilize probability matrix
    classLen = len(trained) #3 classes
    probability = np.zeros((X_test.shape[0], len(trained))) #l = number of samples in x_test per class
    probTrain = np.zeros((X_train.shape[0], classLen))

    #loop through index and value of classifiers to get probability
    for i, mdl in enumerate(trained):
        #predict_proba returns odds the data in x_test belong to each class (0or1)
        probability[:,i] = mdl.predict_proba(X_test)[:, 1] #want positive class (1)
        probTrain[:, i] = mdl.predict_proba(X_train)[:, 1]

    #get class with highest probability (which column/class has highest chance)
    prediction = np.argmax(probability, axis=1) #axis = 1 gives probability of the positive class
    predTrain = np.argmax(probTrain, axis=1)

    #testing accuracy
    correctPreds = 0
    for i in range(0, len(y_test)):
        prediction[i] = prediction[i]+1
        if prediction[i] == y_test[i]:
            correctPreds = correctPreds + 1
    testingAcc = (correctPreds/len(prediction))*100
    print("Testing Accuracy: ", testingAcc, "%")

    #training accuracy    
    correctPredsTrain = 0 
    for i in range(0, len(y_train)):
        predTrain[i] = predTrain[i] +1
        if predTrain[i] == y_train[i]:
            correctPredsTrain = correctPredsTrain + 1
    trainingAcc = (correctPredsTrain/len(predTrain))*100
    print("Training Accuracy: ", trainingAcc, "%")   

    #printing table of accuracies
    accuracies = [
      [testingAcc, trainingAcc]
    ]
    head = ["Testing Accuracy (%)", "Training Accuracy (%)"]
    print(tabulate.tabulate(accuracies, headers=head, tablefmt="grid")) 

pred = logReg_multi(X1, y, xtest1)
