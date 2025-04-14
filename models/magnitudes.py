"""
Different methods of calculating earthquake magnitudes
"""

import numpy as np
import xarray as xr

def mag_covariate_split(cov, b_low, b_high, split_location, transition_fraction=0.05, scale=None):

    out = xr.Dataset()

    if scale is None:
        scale = transition_fraction * (cov.max() - cov.min())
        out.attrs['scale'] = scale.values

    out['b'] = 0.5 * ((b_high + b_low) + (b_high - b_low) * np.tanh((cov - split_location) / scale))
    out['zeta'] = 0

    return out

def mag_covariate_linear(cov, b0, b_slope):

    out = xr.Dataset()

    out['b'] = (b0 - b_slope * cov).clip(min=0.1)
    out['zeta'] = 0

    return out