"""
Different methods of calculating stress through time in a depleting gas reservoir
(hopefully not restricted to grid based solutions)
"""

import numpy as np
import xarray as xr
import scipy.ndimage.filters as filt
import pandas as pd
from chaintools.chaintools import tools_grid as go


def _get_unsmoothed_gradient(faults, rmax, xlin, ylin):
    """
    Calculate the gradient grid for a single value of rmax

    Parameters
    ----------
    faults : xarray.Dataset
        Ungridded faults
    rmax: float
        Maximum throw/thickness ratio allowed (see Bourne and Oates 2017)
    xlin: numpy.ndarray
        Definition of spatial grid (x)
    ylin: numpy.ndarray
        Definition of spatial grid (y)

    Returns
    -------
    out: xarray.Dataarray
        Gradient grid (based on xlin, ylin) for the single rmax value
    """

    delta_x, delta_y = np.min(np.diff(xlin)), np.min(np.diff(ylin))

    f_x, f_y = faults.x, faults.y
    grad, thickness = faults.grad, faults.thickness
    assert delta_x == delta_y, "Unequal gridding not supported"
    xbins = np.linspace(xlin.min() - delta_x / 2, xlin.max() + delta_x / 2, len(xlin) + 1)
    ybins = np.linspace(ylin.min() - delta_y / 2, ylin.max() + delta_y / 2, len(ylin) + 1)

    rm_filter = faults.r > rmax
    length = faults.length.copy()
    length[rm_filter] = 0.0

    # Assign weighted gradients
    grad_grid, _, _ = np.histogram2d(f_y, f_x, bins=[ybins, xbins], weights=grad * length * thickness, density=False)
    rho_grid, _, _ = np.histogram2d(f_y, f_x, bins=[ybins, xbins], weights=length * thickness, density=False)
    gradient_grid = xr.DataArray(grad_grid, coords={"y": ylin, "x": xlin}).assign_coords({"rmax": rmax})
    density_grid = xr.DataArray(rho_grid, coords={"y": ylin, "x": xlin}).assign_coords({"rmax": rmax})
    gradient_grid = (gradient_grid / density_grid).fillna(0)

    grad = xr.Dataset({"gradient": gradient_grid, "density": density_grid})

    return grad


def _get_gradient(faults, rmax, xlin, ylin):
    """
    Calculate the gradient grid for a given rmax range

    Parameters
    ----------
    faults : xarray.Dataset
        Ungridded faults
    rmax: xarray.Dataarray
        Maximum throw/thickness ratio allowed (see Bourne and Oates 2017)
    xlin: numpy.ndarray
        Definition of spatial grid (x)
    ylin: numpy.ndarray
        Definition of spatial grid (y)

    Returns
    -------
    out: xarray.Dataarray
        Gradient grid (based on xlin, ylin) for each rmax value
    """
    # Loop appears to be inevitable unfortunately, due to the nature of rmax parameter
    gradients = []
    for rmax_choice in rmax:
        gradients.append(_get_unsmoothed_gradient(faults, rmax_choice, xlin, ylin))
    gradients = xr.concat(gradients, dim="rmax")
    return gradients


def _perform_xarray_smoothing(dataarray, kernel_lengths, dims, mode="constant"):
    """
    Helper function to perform guassian smoothing on xarray dataarray

    Parameters
    ----------
    dataarray : xarray DataArray
        Input dataarray to perform gaussian convolution on
    kernel_lengths : xarray DataArray
        The kernel lengths to be used
    dims : list
        The dimensions to smooth

    Returns
    -------
    smoothed_results : xarray DataArray
        The array after convolution with gaussian kernel
    """
    assert set(dims).issubset(dataarray.dims), "Dimensions must be present in DataArray"

    # Gaussian filter does not allow broadcasting, have to loop
    smoothed_results = []
    for kernel_length in kernel_lengths:
        sizes = {d: float(kernel_length / np.min(np.diff(dataarray[d]))) for d in dims}
        kernel = [sizes[k] if k in sizes else 0 for k in dataarray.dims]
        smoothed = xr.zeros_like(dataarray)
        smoothed.data = filt.gaussian_filter(dataarray.data, sigma=kernel, mode=mode)
        smoothed_results.append(smoothed)
    smoothed_results = xr.concat(smoothed_results, dim="sigma")
    smoothed_results = smoothed_results.assign_coords({"sigma": kernel_lengths})

    return smoothed_results


def _perform_normalization(dataarray, target, sum_dims=None, norm_value=0.00121376224017666):
    """
    Normalizes a DataArray by a target value.
    The standard value of the target is 0.00121376224017666. When used with a target of time=1995
    and sum_dims='x' and 'y', the result is backwards compatible with older code versions

    Parameters
    ----------
    dataarray : xarray.Dataarray
        Input data array to be normalized
    target : dict
        Dimension in dataarray with associated target value
    sum_dims : list
        List of dimensions to sum over to get target value
    norm_value: float
        Value to use for normalization

    Returns
    -------
    dataarray: xarray.DataArray
        Normalized data array
    """
    normalization = norm_value / dataarray.sel(target, method="nearest").sum(dim=sum_dims)

    return dataarray * normalization


def bo_2017_stress(pressure, compaction_coef, faults, hs_exp, sigma, rmax, poisson_ratio=0.2):
    """
    Calculate a grid based smoothed stress based on Bourne and Oates 2017

    Parameters
    ----------
    pressure : xarray.Dataarray
        Reservoir pressure grid
    compaction_coef : xarray.Dataarray
        Linear compaction coefficients
    faults : xarray.Dataset
        Ungridded faults
    hs_exp: float/list/array
        Elastic constant exponent (see Bourne and Oates 2017)
    sigma: float/list/array
        Smoothing kernel length (see Bourne and Oates 2017)
    rmax: float/list/array
        Maximum throw/thickness ratio allowed (see Bourne and Oates 2017)
    poisson_ratio: float/list/array
        Poisson ratio (see Bourne and Oates 2017)

    Returns
    -------
    out: xarray.Dataset
        Dataset containing the Coulomb stress change (dcs), density, gradient (unsmoothed) and vertical strain (ezz)
    """
    # Sanitize input variables
    rmax = go.make_xarray_based("rmax", rmax)
    hs_exp = go.make_xarray_based("hs_exp", hs_exp)
    sigma = go.make_xarray_based("sigma", sigma)

    pressure = pressure * 0.1  # bar to MPa
    delta_p = pressure.isel(time=0) - pressure
    ezz = delta_p * compaction_coef

    grad = _get_gradient(faults, rmax, ezz.x, ezz.y)
    hs = 10**hs_exp
    dcs = grad.gradient * ezz * (1 - 2 * poisson_ratio) / (2 - 2 * poisson_ratio) * (1 / (ezz / delta_p + 1 / hs))
    dcs.values = np.nan_to_num(dcs.values).clip(min=0)

    # Smooth the stresses and densities
    dcs = _perform_xarray_smoothing(dcs, sigma, ["x", "y"])
    density = _perform_xarray_smoothing(grad.density, sigma, ["x", "y"])

    # Normalize for backwards consistency and to ensure that rate and magnitude parameters do not
    # have to change by orders of magnitude, just to accomodate a linear scaling of the stress
    dcs = _perform_normalization(dcs, {"time": pd.to_datetime(1995, format="%Y")}, ["x", "y"])

    # Gather all information in final dataset
    out = xr.Dataset({"dcs": dcs, "density": density, "gradient": grad.gradient, "ezz": ezz})

    return out
