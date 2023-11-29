import numpy as np

def computeCost(X, y, theta):
    # # X and y are not scalars they are matrices
    # # set value of m to number of training samples
    m = len(X)
    # #initialize h 
    h = [[0] for _ in range(m)]
    sqSum = 0
    #set value of h to h = theta1*x + theta0
    h = np.dot(X, theta)
    #subtract while summing
    temp = h - y
    sqSum = np.dot(temp.T, temp)
    # #multiply by 1/2m
    J = (1/(2*m))*(sqSum)
    return J
