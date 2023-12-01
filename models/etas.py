"""
ETAS (Epidemic Type Aftershock Sequence) functions
"""

import math
import numpy as np
import xarray as xr


def get_etas_rates(catalogue, etas_k=0, etas_a=0, etas_p=1.35, etas_q=3.16, etas_c=0.3 / 365.25, etas_d=4.0e6):
    """
    Calculate the expected number of aftershocks and the aftershock rates at the event locations in the catalogue

    Parameters
    ----------
    catalogue : xarray Dataset
        The earthquake catalogue
    etas_k: float or xarray DataArray 
        k parameter for ETAS formulation
    etas_a: float or xarray DataArray 
        a parameter for ETAS formulation
    etas_p: float or xarray DataArray 
        p parameter for ETAS formulation
    etas_q: float or xarray DataArray 
        q parameter for ETAS formulation
    etas_c: float or xarray DataArray 
        c parameter for ETAS formulation 
    etas_d: float or xarray DataArray 
        d parameter for ETAS formulation

    Returns
    -------
     etas_rates : xarray Dataset
        The expected number of aftershocks (integral) and the aftershock rates at the event 
        locations in the catalogue (local rates)
    """

    etas_term1 = (etas_p - 1.0) / etas_c
    etas_term2 = np.tril((1.0 + catalogue["int_e_time"] / etas_c) ** (-etas_p), -1)
    etas_term3 = (etas_q - 1.0) / (math.pi * etas_d)
    etas_term4 = np.tril((1.0 + catalogue["r2"] / etas_d) ** (-etas_q), -1)
    temp = np.exp(etas_a * catalogue["diff_mag"])
    if isinstance(etas_a, xr.DataArray) and len(etas_a) > 1:
        for i, a_val in enumerate(temp.etas_a):
            temp[{"etas_a": i}] = np.tril(temp.sel(etas_a=a_val), -1)
    else:
        temp.values = np.tril(temp, -1)
    etas_term5 = etas_k * temp
    etas_local_term = (etas_term1 * etas_term2 * etas_term3 * etas_term4 * etas_term5).sum(dim="dim_1")
    etas_integral = etas_k * (np.exp(etas_a * catalogue["diff_mag"][-1, :])).sum(dim="dim_1")

    # Drop all dims except etas_params and event dimension
    available_dims = list(etas_integral.coords.keys())
    keep_dims = list(etas_local_term.dims)
    drop_dims = [d for d in available_dims if d not in keep_dims]
    etas_integral = etas_integral.drop_vars(drop_dims)
    etas_local_term = etas_local_term.drop_vars(drop_dims)

    etas_rates = xr.Dataset({"integral": etas_integral, "local_term": etas_local_term})
    return etas_rates
