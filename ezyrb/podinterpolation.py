"""
Module for the generation of the reduced space by using the Proper Orthogonal
Decomposition Interpolation
"""

from ezyrb.parametricspace import ParametricSpace
import numpy as np
import matplotlib.pyplot as plt

class PODInterpolation(ParametricSpace):
    """
    Documentation

    :cvar numpy.ndarray _pod_basis: basis extracted from the proper orthogonal
        decomposition.
    :cvar object _interpolator: interpolating object for the pod basis
        interpolation
    """

    def __init__(self):

        self._pod_basis = None
        self._interpolator = None

    @property
    def pod_basis(self):
        """
        The basis found by proper orthogonal decomposition.

        :type: numpy.ndarray
        """
        return self._pod_basis

    @property
    def interpolator(self):
        """
        The multidimensional interpolator that combines the saved snapshots for
        the parametric space creation.

        :type: object
        """
        return self._interpolator

    def generate(self, points, snapshots, truncate, interpolator, smoothness):
        """
        Generate the reduced space using the proper orthogonal decomposition
        interpolation: the matrix that contains the `snapshots` computed for
        the `points` is decomposed using SVD tecnique to obtain the POD basis.
        The POD basis are used to compute the modal coefficients. The
        interpolator combines this coefficients.

        :param Snapshots snapshots: the snapshots.
        :param Points points: the parametric points where snapshots were
            computed.
        :param object interpolator: the interpolator used to interpolate the
            coefficients.
        """

        if truncate is None:
            self.truncate = snapshots.weighted.shape[1]
        else:
            self.truncate = int(truncate)

        eig_vec, sing_values, right = np.linalg.svd(snapshots.weighted, full_matrices=False)

        #np.savetxt('sing_values.txt',sing_values)

        #sing_values = sing_values/sing_values[0]

        #plt.plot(np.log10(sing_values[:80]),'.')
        #plt.plot(sing_values[1:],'.')
        #plt.ylim([-3.,0])
        #plt.savefig('eigen.png')
        #plt.close()

        self._pod_basis = np.sqrt(snapshots.weights) * eig_vec
        coefs = self._pod_basis[:,:self.truncate].T.dot(snapshots.weighted)
        if(smoothness==0):
            self._interpolator = interpolator(points.values.T, coefs.T)
        else:
            self._interpolator = interpolator(points.values.T, coefs.T, smoothness=smoothness)

    def __call__(self, value):
        """
        Project a new parametric point onto the reduced space and a new
        approximated solution is provided.

        :param numpy.ndarray value: the new parametric point
        """


        return self._pod_basis[:,:self.truncate].dot(self._interpolator(value).T)

    @staticmethod
    def loo_error(points, snapshots, func=np.linalg.norm):
        """
        Compute the error for each parametric point as projection of the
        snapshot onto the POD basis with a leave-one-out (loo) strategy.

        :param Points points: the points where snapshots were computed.
        :param Snapshots snapshots: the snapshots.
        :param function func: the function to estimate error; default is the
            :func:`numpy.linalg.norm`.
        """
        loo_error = np.zeros(points.size)

        for j in np.arange(points.size):

            remaining_index = list(range(j)) + list(range(j + 1, points.size))
            remaining_snaps = snapshots[remaining_index]

            eigvec = np.linalg.svd(
                remaining_snaps.weighted, full_matrices=False)[0]

            loo_basis = np.sqrt(remaining_snaps.weights) * eigvec

            projection = np.sum(
                np.array([
                    np.dot(snapshots[j].weighted, basis) * basis
                    for basis in loo_basis.T
                ]),
                axis=0)

            error = (snapshots[j].values - projection) * snapshots.weights
            loo_error[j] = func(error) / func(snapshots[0].values)

        return loo_error
