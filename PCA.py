"""
Description: Linear Principle Component Analysis
"""
from numpy import *
from numpy.linalg import *


'''
PCA takes an array of points, each row is a single point. and reduce the dimension
    of the points to k-D.
    return points of k-D.
'''


def pca(x, k):
    m = x.shape[0]
    c = dot(transpose(x), x) / m  # coefficient
    [u, s, v] = svd(c)
    p = u[:, 0:k]
    y = dot(x, p)
    return y