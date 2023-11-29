import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import statistics as stat
import math

def sigmoid(z):
    # calc exponential
    temp = (1+np.exp(-z))
    g = 1/temp
    return g