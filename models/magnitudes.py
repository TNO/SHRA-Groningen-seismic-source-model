"""
Different methods of calculating earthquake magnitudes
"""

import numpy as np
import xarray as xr


def mag_tanh_bval(stress, theta0, theta1, theta2):
    """
    Calculate the bvalue based on the tanh magnitude model.

    Parameters
    ----------
    stress: xarray.DataArray
        The Coulomb stress change (dcs)
    theta0: float or xarray DataArray
        Model parameter
    theta1: float or xarray DataArray
        Model parameter
    theta2: float or xarray DataArray
        Model parameter

    Returns
    -------
    fmd_terms: xarray.DataSet
        Terms for the log-likelihood calculation
    """

    b = (theta0 + theta1 * (1. - np.tanh(theta2 * stress))).clip(max=3.0)
    zeta = 0

    fmd_terms = xr.Dataset({'b': b,
                            'zeta': zeta})

    return fmd_terms
