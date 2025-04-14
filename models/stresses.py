"""
Different methods of calculating stress through time in a depleting gas reservoir
(hopefully not restricted to grid based solutions)
"""

import numpy as np
import xarray as xr
import pandas as pd
import tqdm
from models.strains import rate_type_isotach_compaction_model
from scipy.ndimage import gaussian_filter

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
    xbins = np.linspace(xlin.min().item() - delta_x / 2, xlin.max().item() + delta_x / 2, len(xlin) + 1)
    ybins = np.linspace(ylin.min().item() - delta_y / 2, ylin.max().item() + delta_y / 2, len(ylin) + 1)

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
        smoothed.data = gaussian_filter(dataarray.data, sigma=kernel, mode=mode)
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


def bo_2017_stress(
        pressure,
        compaction_coef,
        reservoir_depth,
        faults,
        model_params,
        out_name=None
):
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
    if out_name is None:
        out_name = 'elastic_plastic'

    # handle parameters:
    default = {
        'poisson_ratio': [0.2],
        'sigma_rate_ref': [3.16e-5],
        'overburden_density': [2400],
        'factor_cm_d': [0.4],
        'factor_cm_ref': [0.8],
        'b_exponent': [0.021]
    }
    model_params = model_params.expand_dims({k: v for k, v in default.items() if k not in model_params.dims})
    if 'M_plastic' not in model_params:
        model_params['M_plastic'] = _get_M_elastic(model_params['poisson_ratio']).rename({'poisson_ratio': 'M_plastic'})
        out_name = 'elastic'

    print('Starting calculation for {} strain'.format(out_name))
    loading_history = _prepare_loading_history(pressure, reservoir_depth, model_params['overburden_density'])
    strain = rate_type_isotach_compaction_model(
        loading_history['effective_vertical_stress'].set_index({'time': 'year_fraction'}),
        compaction_coef,
        model_params,
        output_rates=True
    )
    H = _get_uniaxial_compaction_modulus(compaction_coef * model_params['factor_cm_d'], model_params['hs_exp'])

    # Get stress for homogeneous thin sheet
    stress = thin_sheet_stress(
        strain_1D=strain,
        H=H,
        params=model_params,
        sigma_v_initial=loading_history['effective_vertical_stress'].isel(time=0, drop=True),
    )
    # get stress CHANGE and revert sign (so that compression = positive)
    stress = -1 * (stress - stress.isel({'time': 0}))['horizontal_stress']

    # return time coordinates
    stress = stress.assign_coords({'time': loading_history.time})
    strain = strain.assign_coords({'time': loading_history.time})

    # Get fault amplifier
    grad = _get_gradient(faults, model_params['rmax'], strain.x, strain.y)

    # Get stress for heterogeneous thin sheet (i.e., change in Coulomb stress)
    dcs = stress * grad.gradient
    dcs.data = np.nan_to_num(dcs.data).clip(min=0)
    print("stresses computed")

    # Smooth the stresses and densities
    dcs = _perform_xarray_smoothing(dcs, model_params['sigma'], ["x", "y"])
    density = _perform_xarray_smoothing(grad.density, model_params['sigma'], ["x", "y"])
    print("stresses smoothed")

    # Normalize for backwards consistency and to ensure that rate and magnitude parameters do not
    # have to change by orders of magnitude, just to accommodate a linear scaling of the stress
    dcs = _perform_normalization(dcs, {"time": pd.to_datetime(1995, format="%Y")}, sum_dims=["x", "y"])

    # Gather all information in final dataset
    out = xr.Dataset({"dcs": dcs, "density": density, "gradient": grad.gradient})
    out = out.assign_attrs({'branch_rate': out_name})

    return out

def _prepare_loading_history(pressure, reservoir_depth, overburden_density, conversion_factor=0.1) -> xr.Dataset:
    """
    Manipulate reservoir pressures to pressure changes and effective vertical stress. Outcome is always in MPa.
    Default conversion factor of pressure is bars to MPa.
    Density and reservoir depth are assumed to be in SI units (kgm3, m).

    :param pressure:
    :param reservoir_depth:
    :param params:
    :param conversion_factor:
    :return:
    """

    pressure = pressure * conversion_factor
    out = xr.Dataset({
        'absolute_pressure': pressure,
        'pressure_change': pressure.isel(time=0) - pressure,
        'effective_vertical_stress': - (overburden_density * 9.81 * reservoir_depth) / 1e6 + pressure
    })
    year_fraction_since_start = (out.time.values - out.time.values[0]) / pd.Timedelta('365.25 days')
    out = out.assign_coords({'year_fraction':('time', year_fraction_since_start)})

    return out

def _prepare_param_M(M, nu, include_elastic_solution=False):

    M_list =[]
    if M is None and include_elastic_solution is False:
        include_elastic_solution = True
        print('CAUTION: M not given, only calculating elastic stress model')
    else:
        M = M.assign_coords({'strain_type': ('M_plastic', np.full(len(M), 'elastic_plastic'))})
        M_list.append(M)

    if include_elastic_solution:
        M_el = _get_M_elastic(nu).rename({'poisson_ratio': 'M_plastic'})
        M_el = M_el.assign_coords({'M_plastic': M_el.values, 'strain_type': ('M_plastic', ['elastic'])})
        M_list.append(M_el)

    return xr.concat(M_list, dim='M_plastic')

def _get_M_elastic(nu):
    """
    See equation 36 in Aben et al., 2025. For derivation, see Appendix B.

    :param nu:
    :return:
    """
    M = 3 * ( ((1 - nu / (1 - nu) ) / (1 + 2 * nu / (1 - nu))) ** 2
              + ((1 - nu / (1 - nu) ) / (1 + 2 * nu / (1 - nu))) ) ** 0.5

    return M

def _get_uniaxial_compaction_modulus(cm, hs_exp):
    """
    Calculation of uniaxial compaction modulus H from compaction coefficient and Hs.
    See Bourne & Oates 2017, section 2.2, or equations 13 & 14 in Aben et al. 2025
    :param cm:
    :param hs_exp:
    :return:
    """
    H = 1 / (cm + 1 / (10 ** hs_exp))

    return H

def thin_sheet_stress(strain_1D: xr.Dataset, H: xr.DataArray, params: xr.Dataset,
                      sigma_v_initial: xr.DataArray):
    """
    Calculates horizontal stress in a geometrically homogeneous thin sheet geometry, based on vertical strain
    rates. Vertical strain rates can be elastic or elastic and plastic.

    :param strain_1D:
    :param H:
    :param params:
    :param sigma_v_initial:
    :param include_elastic_solution:
    :return:
    """
    # prepare input time vectors
    dt = strain_1D.time.values[1:] - strain_1D.time.values[:-1]
    dstrain_1D_e = strain_1D['elastic_strain_rate']
    dstrain_1D_i = strain_1D['inelastic_strain_rate']
    dalpha_pf = (xr.DataArray(dt, dims='time') * dstrain_1D_e[1:]).cumsum(dim='time') * H

    # parameters and constants
    nu = params['poisson_ratio']
    M = params['M_plastic']

    # ------- set initial value for horizontal effective stress and strains -------
    k_0_eff = nu / (1 - nu)
    alpha_pf = sigma_v_initial
    sh_eff = sigma_v_initial * k_0_eff
    sh_tot = sh_eff

    # -------
    # prepare output
    list_sh = [sh_tot]
    list_sh_eff = [sh_eff]

    # let's go!
    desc = 'calculating horizontal stress change'
    for i in tqdm.tqdm(range(len(dt)), desc=desc):

        P = sh_eff * (2 / 3) + alpha_pf * (1 / 3)
        Q = np.abs(alpha_pf - sh_eff)

        # update vertical stress w biot pore pressure correction
        alpha_pf = sigma_v_initial + dalpha_pf.isel({'time': i}, drop=True)

        dot_eps_e = dstrain_1D_e.isel({'time': i}, drop=True)
        dot_eps_i = dstrain_1D_i.isel({'time': i}, drop=True)
        dPs_dP = 1 - Q ** 2 / (M ** 2 * P ** 2)
        dPs_dQ = 2 * Q / (M ** 2 * P)

        dotSigma_h_eff = (
                H / (1 - nu) * (
                dot_eps_e * nu +
                dot_eps_i * (0.5 * dPs_dQ + 1 / 3 * dPs_dP) / (dPs_dQ - 1 / 3 * dPs_dP)
        ))

        dotSigma_h = - (
                 H * (1 - 2 * nu) / (1 - nu) * (
                dot_eps_i * (0.5 * dPs_dQ + 1 / 3 * dPs_dP) / (dPs_dQ - 1 / 3 * dPs_dP)
                - dot_eps_e
        ))

        sh_eff = sh_eff + dotSigma_h_eff * dt[i]
        sh_tot = sh_tot + dotSigma_h * dt[i]
        list_sh.append(sh_tot)
        list_sh_eff.append(sh_eff)

    out = xr.Dataset({
        'horizontal_stress': xr.concat(list_sh, dim='time'),
        'horizontal_effective_stress': xr.concat(list_sh_eff, dim='time')
    }).assign_coords({'time':strain_1D.time})

    return out

