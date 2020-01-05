"""
Module for Radial Basis Function Interpolation
"""
from scipy.interpolate import Rbf
import numpy as np
import GPy

class GPInterpolator(object):
    """
    Multidimensional interpolator using Radial Basis Function.

    :param array_like points: the coordinates of the points.
    :param array_like values: the values in the points.
    :param float radius: the radius used in the basis functions. Default is 1.0.
    :param norm: The function that returns the distance between two points. It
        has to be a function that take as input a vector and return a float
        number.  Default is 'euclidean'.
    :type norm: str or callable
    :param callable basis: the basis function.

    :cvar array_like points: the coordinates of the points.
    :cvar float radius: the radius used in the basis functions. Default is 1.0.
    :cvar callable basis: the basis function.
    :cvar numpy.ndarray weights: the weights matrix.
    """

    def __init__(self, points, values):
        
        k = GPy.kern.RBF(input_dim=points.shape[1])
        self.m=GPy.models.GPRegression(points,values,k)

        #self.m.optimize()
        self.m.optimize_restarts(num_restarts = 10)


    def __call__(self, new_points):
        """
        Evaluate interpolator at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        return self.m.predict(new_points)[0]