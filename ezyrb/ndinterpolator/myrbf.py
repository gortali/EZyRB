"""
Module for Radial Basis Function Interpolation
"""
from scipy.interpolate import Rbf
import numpy as np

class myRBFInterpolator(object):
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

    def __init__(self, points, values, smoothness=0.):

        argument = np.hstack([points, values])

        self.rbfs = []
        for i in range(points.shape[1], points.shape[1]+values.shape[1]):
            a = np.zeros(shape=argument.shape[1], dtype=bool)
            a[:points.shape[1]] = True
            a[i] = True
            self.rbfs.append(Rbf(*argument[:, a].T, smooth=smoothness))



    def __call__(self, new_points):
        """
        Evaluate interpolator at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """

        return np.array([rbf(*new_points.T) for rbf in self.rbfs]).T
