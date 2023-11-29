import scipy
import numpy as np

####Regularization###

#computing the closed-form solution using normal equation and regularization
def Reg_normalEqn(X_train, y_train, lambda1):
    #get length of columns
    dim = X_train.shape[1]
    #get transpose of X
    X_transpose = X_train.T
    #mult transpose and x
    product1 = np.dot(X_transpose, X_train)
    #identity matrix with top corner term = 0 for bias feature
    identityMatrix = np.eye(dim)
    identityMatrix[0][0] = 0
    #get lambda product
    product2 = np.dot(lambda1, identityMatrix)
    #get inverse of the products added
    inverse = np.linalg.pinv(product1 + product2)
    #get another product
    # temp = np.dot(X_transpose, y_train)
    # thetaEst = np.dot(inverse, temp)
    thetaEst = inverse@X_transpose@y_train

    print("Theta Estimate: ")
    # print(thetaEst)
    return thetaEst

