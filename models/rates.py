"""
Different methods of calculating seismicity through time in a depleting gas reservoir
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from scipy.ndimage import convolve1d
from tools.catalogue_tools import filter_attr_str2list, covariate_at_event
from tools.polygon_tools import get_polygon_gdf_from_points
from models.etas import get_etas_spatial_decay_kernel
from models.magnitudes import mag_covariate_split, mag_covariate_linear
from chaintools.chaintools import tools_grid as go
from chaintools.chaintools import tools_xarray as xf
from chaintools.chaintools import tools_geometry as tg
from typing import Union, List

def bo_2017_nr_events_in_time_interval(stress, theta0, theta1, start_time, end_time, spatial_mask):
    """
    Calculate number of earthquakes in the full area in a time period, following the Bourne and Oates (2017) method.

    Parameters
    ----------
    stress: xarray.Dataset
        Dataset containing the Coulomb stress change (dcs), density, gradient (unsmoothed) and vertical strain (ezz)
    theta0: float or xarray DataArray
        Rate prefactor (see Bourne and Oates 2017)
    theta1: float or xarray DataArray
        Stress dependence (see Bourne and Oates 2017)
    start_time: pandas.datetime
        Start of the period of interest
    end_time: pandas.datetime
        End of the period of interest
    spatial_mask: xarray.Datarray
        Used to 'turn off' grid cells that should not be considered in the spatial integral
    Returns
    -------
    nr_events: xarray.DataArray
        The total number of events in the time period
    """
    stress_start = stress.interp(time=start_time)
    stress_end = stress.interp(time=end_time)
    cell_area = (stress.x[1] - stress.x[0]) * (stress.y[1] - stress.y[0])
    rho = stress["density"] * spatial_mask

    nr_events = (
        cell_area
        * np.exp(theta0)
        * (rho * (np.exp(theta1 * stress_end.dcs) - np.exp(theta1 * stress_start.dcs))).sum(dim=["x", "y"])
    )
    return nr_events


def bo_2017_event_rates_at_observations(catalogue, stress, theta0, theta1):
    """
    Calculate the rate of change of the number of earthquakes in a time period

    Parameters
    ----------
    catalogue: xarray.Dataset
        Dataset containing earthquake information
    stress: xarray.Dataset
        Dataset containing the Coulomb stress change (dcs), density, gradient (unsmoothed) and vertical strain (ezz)
    theta0: float or xarray DataArray
        Rate prefactor (see Bourne and Oates 2017)
    theta1: float or xarray DataArray
        Stress dependence (see Bourne and Oates 2017)

    Returns
    -------
    rates_at_obs: xarray DataArray
        Event rate (per square meter per year) at the location of each observed event
    """

    dcs = covariate_at_event(stress["dcs"], catalogue)
    # We clip the rates at a very small but postive number to prevent likelihoods going to zero when event occur at
    # locations of increasing pressure.
    dcs_rate = covariate_at_event(stress["dcs"], catalogue, rate=True, rate_clip=1e-50)
    rho = covariate_at_event(stress["density"], catalogue)

    rates_at_obs = rho * theta1 * dcs_rate * np.exp(theta0 + theta1 * dcs)

    return rates_at_obs


def bo_2017_ll_terms(stress, catalogue, theta0, theta1):
    """
    Calculate the log-likelhood terms for the Bourne and Oates (2017) model.

    Parameters
    ----------
    stress: xarray.Dataset
        Dataset containing the Coulomb stress change (dcs), density, gradient (unsmoothed) and vertical strain (ezz)
    catalogue: xarray.Dataset
        Dataset containing earthquake magnitudes and earthquake locations
    theta0: float or xarray DataArray
        Rate prefactor (see Bourne and Oates 2017)
    theta1: float or xarray DataArray
        Stress dependence (see Bourne and Oates 2017)

    Returns
    -------
    bo_rates: xarray.DataSet
        Terms for the log-likelihood calculation
    """

    # Extract earthquake catalogue period which we need to predict rates for
    start_time, end_time = filter_attr_str2list(catalogue.date_filter)
    start_time = pd.to_datetime(start_time, format=r"%Y%m%d")
    end_time = pd.to_datetime(end_time, format=r"%Y%m%d") + pd.DateOffset(days=1) - pd.DateOffset(seconds=1)

    # If the catalogue was filtered spatially, ensure that the forecasted rates are only considered in the same area
    polygon = get_polygon_gdf_from_points(catalogue.location_filter_x, catalogue.location_filter_y)
    spatial_mask = compute_overlap_fraction(stress, polygon)

    # Calculate the log-likelhood terms for background rate
    total_events = bo_2017_nr_events_in_time_interval(stress, theta0, theta1, start_time, end_time, spatial_mask)
    rates_at_obs = bo_2017_event_rates_at_observations(catalogue, stress, theta0, theta1)

    bo_rates = xr.Dataset({"integral": total_events, "local_term": rates_at_obs})

    return bo_rates

def compute_overlap_fraction(stress, polygon):
    polygon_xr = xr.DataArray.from_series(polygon.geometry)
    radius = 0.5 * (stress.x[1] - stress.x[0])
    spatial_mask = tg.xr_cell_polygon_overlap_fraction(
        stress, polygon_xr, radius, cap_style="square"
    ).isel(index=0)
    
    return spatial_mask


def bo_2017_forecast(ar_posterior, mag_posterior, stress_posterior, stress, mags, mmax, rate_bool,
                     full_bool, polygon):
    """
    Calculate the seismicity forecast for the Bourne and Oates (2017) model.

    Parameters
    ----------
    ar_posterior: xarray.DataArray
        Array containing the posterior probability of the activity rate model parameters
    mag_posterior: xarray.DataArray
        Array containing the posterior probability of the magnitude model parameters
    stress_posterior: xarray.DataArray
        Array containing the posterior probability of the stress model parameters
    stress: xarray.Dataset
        Data set containing Coulomb stress change (dcs), density, gradient (unsmoothed) and vertical strain (ezz)
    mags: xarray.DataArray
        Array containing the magnitude values on which to calculate the forecast
    mmax: xarray.DataArray
        Array containing the values of the maximum magnitude
    rate_bool: np.array of boolean values
        Array indicating which part of the time dimension in 'stress' should be used for a rate forecast
    full_bool: np.array of boolean values
        Array indicating which part of the time dimension in 'stress' should be used for a rate forecast


    Returns
    -------
    event_rate_forecast: xarray.DataArray
        Posterior predictive forecast of the seismicity rate
    full_forecast: xarray.DataArray
        Posterior predictive forecast of the seismicity rate and magnitudes
    nr_event_pmf: xarray.DataArray
        Probability mass function of the number of events per time step
    """
    erf_out = []
    ff_out = []
    nr_e_pmf_out = []

    spatial_mask = compute_overlap_fraction(stress, polygon)
    spatial_etas_kernel = get_etas_spatial_decay_kernel()

    for mf_model in ar_posterior.data_vars:
        # select per model
        stress_posterior_flat = stress_posterior[mf_model].stack(flat=[...])

        event_rate_forecast, full_forecast, nr_event_pmf = 0, 0, 0

        # forecast per stress_posterior member
        desc = 'forecasting members of model {}'.format(mf_model)
        for stress_member in tqdm(stress_posterior_flat.flat, desc=desc):
            # select posterior members and covariate members
            selection_dict = {a: stress_member[a].item() for a in stress_posterior[mf_model].dims}
            member_cov_ds = stress.sel(**selection_dict)
            member_ar_posterior = ar_posterior[mf_model].sel(**selection_dict)
            if mf_model == 'stress_tanh' or mf_model == 'linear_stress':
                member_mag_posterior = mag_posterior[mf_model].sel(**selection_dict)
                member_mag_cov = member_cov_ds.dcs
            elif mf_model == 'split_thickness':
                member_mag_posterior = mag_posterior[mf_model]
                member_mag_cov = member_cov_ds.reservoir_thickness
            else:
                raise UserWarning(f"Forecasting with magnitude model {mag_posterior.mag_model} not supported")

            # rescale probabilities
            member_ar_posterior /= member_ar_posterior.sum()
            member_mag_posterior /= member_mag_posterior.sum()

            # forecast member
            member_event_rate, member_forecast, member_nr_event_pmf = _single_stress_forecast(
                member_ar_posterior,
                member_mag_posterior,
                member_cov_ds.dcs,
                member_mag_cov,
                member_cov_ds.density,
                mags,
                mmax,
                rate_bool,
                full_bool,
                spatial_etas_kernel,
                spatial_mask
            )

            member_probability = stress_posterior_flat.sel(flat=stress_member)
            event_rate_forecast += member_probability * member_event_rate
            full_forecast += member_probability * member_forecast
            nr_event_pmf += member_probability * member_nr_event_pmf

        model_version_dict = {'branch_mm':np.atleast_1d(mf_model)}
        erf_out.append(xf.xr_clean_coords(event_rate_forecast.expand_dims(model_version_dict)))
        ff_out.append(xf.xr_clean_coords(full_forecast.expand_dims(model_version_dict)))
        nr_e_pmf_out.append(xf.xr_clean_coords(nr_event_pmf.expand_dims(model_version_dict)))

    return xr.combine_by_coords(erf_out), xr.combine_by_coords(ff_out), xr.combine_by_coords(nr_e_pmf_out)


def _single_stress_forecast(
        ar_posterior,
        mag_posterior,
        ar_cov,
        mag_cov,
        density,
        mags,
        mmax,
        rate_bool,
        full_bool,
        spatial_etas_kernel,
        spatial_mask
):
    """
    Calculate the seismicity forecast for the Bourne and Oates (2017) model for a single stress realization.
    We create the forecast in two steps. First, we create a lookup table for the total event rate and magnitude
    distribtion as a function of coulomb stress. The second step simply convolves the lookup table with the calculated
    coulomb stress fields to obtain the forecast

    Two forecasts are generated: A full forecast (space-time-magnitude) and a rate forecast (space-time).

    Spatial contour filtering is only applied to rates (these are used for total event rate/count figures).

    Number of event uncertainties, including aftershocks, are provided for the rate forecast only.

    Parameters
    ----------
    ar_posterior: xarray.DataArray
        Array containing the posterior probability of the activity rate model parameters
    mag_posterior: xarray.DataArray
        Array containing the posterior probability of the magnitude model parameters
    ar_cov: xarray.DataArray
        Array containing the Coulomb stress change (dcs) used for the activity rate model
    mag_cov: xarray.DataArray
        Array containing the covariate for the magnitude model
    density: xarray.DataArray
        Array containing the fault density
    mags: xarray.DataArray
        Array containing the magnitude values on which to calculate the forecast
    mmax: xarray.DataArray
        Array containing the values of the maximum magnitude
    rate_bool: np.array of boolean values
        Array indicating which part of the time dimension in 'stress' should be used for a rate forecast
    full_bool: np.array of boolean values
        Array indicating which part of the time dimension in 'stress' should be used for a rate forecast
    spatial_etas_kernel: np.array
        Array with the spatial smoothing of aftershocks
    spatial_mask: xarray.DataArray
        Array containing the spatial mask with values between 0 and 1 or bools.

    Returns
    -------
    event_rate_forecast: xarray.DataArray
        Posterior predictive forecast of the seismicity rate
    full_forecast: xarray.DataArray
        Posterior predictive forecast of the seismicity rate and magnitudes
    nr_event_pmf: xarray.DataArray
        Probability mass function of the number of events per time step
    """

    # Step 1: Create the lookup table (see doc folder for the derivation of the functions below)
    lookup_table, mainshock_lookup_table = create_lookup_table(ar_cov, mag_cov, mags, mmax, ar_posterior, mag_posterior)

    # Step 2: Convolve stresses with lookup table to obtain the forecasts
    # Step 2a: Rate forecast
    rate_forecast, rate_ms_forecast = convolve_lookup_table(
        lookup_table.isel(magnitude=0, branch_mmax=-1),
        mainshock_lookup_table.isel(magnitude=0, branch_mmax=-1),
        ar_cov,
        mag_cov,
        density,
        rate_bool
    )
    rate_forecast = _apply_etas_spatial_decay(rate_forecast, rate_ms_forecast, spatial_etas_kernel)
    rate_forecast, rate_ms_forecast = rate_forecast * spatial_mask, rate_ms_forecast * spatial_mask

    nr_event_pmf = _get_nr_event_pmf(rate_ms_forecast, rate_forecast)

    # Step 2b: Full forecast
    full_forecast, ms_forecast = convolve_lookup_table(
        lookup_table,
        mainshock_lookup_table,
        ar_cov,
        mag_cov,
        density,
        full_bool
    )
    full_forecast = _apply_etas_spatial_decay(full_forecast, ms_forecast, spatial_etas_kernel)

    return rate_forecast, full_forecast, nr_event_pmf

def _apply_etas_spatial_decay(full_forecast, ms_forecast, spatial_etas_kernel):
    """

    Parameters
    ----------
    full_forecast
    ms_forecast
    spatial_etas_kernel

    Returns
    -------

    """

    aftershock_forecast = (full_forecast - ms_forecast).fillna(0)
    aftershock_forecast = convolve_spatial_kernel(aftershock_forecast, spatial_etas_kernel, dim=["x", "y"])
    full_forecast = ms_forecast.fillna(0) + aftershock_forecast

    return full_forecast

def _get_nr_event_pmf(ms_forecast: xr.DataArray, full_forecast: xr.DataArray):
    """
    Using the rate forecast and the mainshock rate forecast, we can retrieve the effective aftershock productivity
    Together with the rate, we can use this to calculate the uncertainty in the number of events, including dispersion
    caused by ETAS

    Parameters
    ----------
    ms_forecast
    full_forecast

    Returns
    -------

    """
    mainshock_rate = ms_forecast.sum(dim=["x", "y"])
    effective_r = 1 - 1 / (full_forecast.sum(dim=["x", "y"]) / mainshock_rate)
    lookup_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "res", "event_uncertainty_with_etas.h5"
    )
    event_uncertainty_lookup = xr.open_dataarray(lookup_path)
    nr_event_pmf = event_uncertainty_lookup.interp(background_rate=mainshock_rate, r=effective_r)

    return nr_event_pmf


def convolve_lookup_table(lookup_table, ms_lookup_table, ar_cov, mag_cov, density, time_bool):
    """
    Do the convolution of the lookup table with the modelled stresses

    Parameters
    ----------
    lookup_table: xarray.DataArray
        Expected integral of the ccfmd w.r.t. the covariate
    ms_lookup_table: xarray.DataArray
        Lookup table if we explicitly include *only* the mainshocks
    ar_cov: xarray.DataArray
        Array containing the Coulomb stress change (dcs) used for the activity rate model
    mag_cov: xarray.DataArray
        Array containing the covariate for the magnitude model
    density: xarray.DataArray
        Fault density
    time_bool: np.array of boolean values
        Array indicating which part of the time dimension in the covariates should be used for the forecast

    Returns
    -------
    full_forecast: xarray.DataArray
        Posterior predictive forecast of the seismicity rate
    ms_forecast: xarray.DataArray
        Posterior predictive forecast of the seismicity rate by *only* mainshocks

    """
    # Get the dcs change per time step and the dcs in the middle of the time step
    delta_stress = ar_cov.diff(dim="time", label="lower")
    mid_stress = 0.5 * (
            ar_cov.isel(time=slice(None, -1)).assign_coords(time=delta_stress.time)
            + ar_cov.isel(time=slice(1, None)).assign_coords(time=delta_stress.time)
    )

    # The last entry of the time_bool array is not used due to the diff in time
    time_bool = time_bool.isel(time=slice(None, -1))

    # Get stresses used for forecast
    d_stress = delta_stress.isel(time=time_bool).clip(min=0.0)
    mid_stress = mid_stress.isel(time=time_bool)

    computation_method = "direct"
    if mag_cov.name == "reservoir_thickness":
        interp_values = {"dcs": mid_stress, "reservoir_thickness": mag_cov}
        if "branch_mmax" in lookup_table.dims:
            computation_method = "serial"
    elif mag_cov.name == "dcs":
        interp_values = {"dcs": mid_stress}
    else:
        raise UserWarning(f"Forecasting with magnitude covariate {mag_cov.mag_model} not supported")

    # get forecasts:
    full_forecast = _convolve(lookup_table, interp_values, computation_method)
    ms_forecast = _convolve(ms_lookup_table, interp_values, computation_method)

    # Scale lookup results with the area of the cell and the fault density within that cell
    cell_area = (ar_cov.x[1] - ar_cov.x[0]) * (ar_cov.y[1] - ar_cov.y[0])
    scale = density * cell_area
    full_forecast = scale * full_forecast * d_stress
    ms_forecast = scale * ms_forecast * d_stress

    return full_forecast, ms_forecast


def convolve_spatial_kernel(da: xr.DataArray, kernel: np.array, dim: Union[str, List], mode='constant'):

    if isinstance(dim, str):
        dim = [dim]

    for d in dim:
        axis_index = da.get_axis_num(d)
        da.values = convolve1d(da, kernel, axis_index, mode=mode)

    return da


def interp_lookup(table, interp_dict):
    return table.interp(interp_dict)

def _convolve(lookup_table, interp_values, computation_method=None):

    # obtain full forecast
    if computation_method == "dask":
        # interpolate by parallel computing over mmax
        lookup_table = lookup_table.chunk({"branch_mmax": 1})
        job = interp_lookup(lookup_table, interp_values)
        full_forecast = job.compute()
        ProgressBar(full_forecast)

    elif computation_method == "serial":
        # interpolate by looping over mmax
        full_forecast = interp_values['dcs'].broadcast_like(lookup_table, exclude=interp_values.keys()).copy()
        dim_order = full_forecast.isel(branch_mmax=0).dims

        for mmax in lookup_table.branch_mmax:
            full_forecast.loc[{"branch_mmax": mmax}] = (
                interp_lookup(lookup_table.sel(branch_mmax=mmax), interp_values)
                .transpose(*dim_order)
            )
    else:
        # interpolate in one go
        full_forecast = interp_lookup(lookup_table, interp_values)

    return full_forecast


def create_lookup_table(ar_cov, mag_cov, mags, mmax, ar_posterior, mag_posterior):
    """
    Calculate the derivative of the ccfmd w.r.t. the covariates

    Parameters
    ----------
    ar_posterior: xarray.DataArray
        Array containing the posterior probability of the activity rate model parameters
    mag_posterior: xarray.DataArray
        Array containing the posterior probability of the magnitude model parameters
    ar_cov: xarray.DataArray
        DataArray containing the covariate for the activity rate model
    mag_cov: xarray.DataArray
        DataArray containing the covariate for the magnitude model
    mags: xarray.DataArray
        Array containing the magnitude values on which to calculate the forecast
    mmax: xarray.DataArray
        Array containing the values of the maximum magnitude

    Returns
    -------
    lookup_table: xarray.DataArray
        Expected integral of the ccfmd w.r.t. the covariate
    mainshock_lookup_table: xarray.DataArray
        The equivalent table if only mainshocks are considered
    """

    # Create discretized stress values to calculate on. We use both bin edges and bin centres, depending on the function
    discr_ar_cov = discretize_variable(ar_cov, start=0)
    if mag_posterior.name == 'split_thickness':
        discr_mag_cov = discretize_variable(mag_cov, start=mag_cov.min().values)
    elif mag_posterior.name == 'stress_tanh' or mag_posterior.name == 'linear_stress':
        discr_mag_cov = discr_ar_cov
    else:
        raise UserWarning(f"Magnitude model {mag_posterior.name} not supported")

    # Calculate the joint probability mass function of the b-value and zeta as a function of the covariate
    b_zeta_pmf = get_joint_pmf_b_zeta(mag_posterior, discr_mag_cov)

    # Calculate the complimentary cumulative magnitude frequency distribution as function of b and zeta
    ccmfd_b_zeta = get_ccmfd_as_function_b_zeta(b_zeta_pmf.b, b_zeta_pmf.zeta, mags, mmax)

    # Calculate the effective ETAS aftershock multiplier (total number of expected aftershocks over all generations
    # for a single main shock)
    aftershock_multiplier = get_aftershock_multiplier(ccmfd_b_zeta, ar_posterior.etas_k, ar_posterior.etas_a)

    # Calculate the earthquake cumulative rate derivative with respect to the covariate (expected number of earthquakes
    # for a unit change in covariate)
    dlambda_dcov = get_rate_derivative_bo2017(ar_posterior.theta0, ar_posterior.theta1, discr_ar_cov)

    # Combine all contributions to the rate in a single lookup table, accounting for the posterior probability of all
    # parameters used
    all_dims = list(
        set(aftershock_multiplier.dims + dlambda_dcov.dims + b_zeta_pmf.dims + ccmfd_b_zeta.dims + ar_posterior.dims)
    )
    keep_dims = ["magnitude", "branch_mmax"] + [ar_cov.name] + [mag_cov.name]
    sum_dims = [d for d in all_dims if d not in keep_dims]

    lookup_table = xr.dot(
        aftershock_multiplier,
        dlambda_dcov,
        b_zeta_pmf,
        ccmfd_b_zeta,
        ar_posterior,
        dim=sum_dims,
        optimize="greedy",
    )

    mainshock_lookup_table = xr.dot(
        xr.ones_like(aftershock_multiplier),
        dlambda_dcov,
        b_zeta_pmf,
        ccmfd_b_zeta,
        ar_posterior,
        dim=sum_dims,
        optimize="greedy",
    )

    return lookup_table, mainshock_lookup_table


def get_rate_derivative_bo2017(theta0, theta1, dcs):
    """
    Calculate the earthquake cumulative rate derivative with respect to the covariate (expected number of earthquakes
    for a unit change in covariate)

    Parameters
    ----------
    theta0: float or xarray DataArray
        Rate prefactor (see Bourne and Oates 2017)
    theta1: float or xarray DataArray
        Stress dependence (see Bourne and Oates 2017)
    dcs: xarray.DataArray
        Coulomb stress change

    Returns
    -------
    dlambda_dcov: xarray.DataArray
        Rate derivative
    """

    return theta1 * np.exp(theta0 + theta1 * dcs)


def get_aftershock_multiplier(ccmfd, etas_k, etas_a, mmin=None):
    """
    Calculate the effective ETAS aftershock multiplier (total number of expected aftershocks over all generations
    for a single main shock)

    Parameters
    ----------
    ccmfd: xarray.DataArray
        Complimentary cumulative magnitude frequency distribution
    etas_k: xarray.DataArray
        ETAS K parameter (activity)
    etas_a: xarray.DataArray
        ETAS a parameter (magnitude-dependence)
    mmin: float (optional)
        The minimum magnitude used in the forecast (defaults to minimum magnitude of ccfmd)

    Returns
    -------
    aftershock_multiplier: xarray.DataArray
        Effective multiplier as function of ETAS parameters and magnitude distribution parameter b and zeta
    """

    if mmin is None:
        mmin = ccmfd.magnitude.min()

    pmf = ccmfd2pmf(ccmfd)
    r = (etas_k * np.exp(etas_a * (pmf.magnitude - mmin)) * pmf).sum(dim="magnitude")
    r.values[np.where(r.values > 0.9)] = 0.9

    aftershock_multiplier = 1.0 / (1.0 - r)

    return aftershock_multiplier


def ccmfd2pmf(ccmfd):
    """
    Converts a Complementary Cumulative Magnitude Frequency Distribution into a Probability Mass Function

    Parameters
    ----------
    ccmfd: xarray.DataArray
        Complimentary cumulative magnitude frequency distribution

    Returns
    -------
    pmf: xarray.DataArray
        Probability mass function of the magnitude frequency distribution


    """
    pmf_mags = ccmfd.magnitude.isel(magnitude=slice(1, -1, 2))
    pmf = ccmfd.isel(magnitude=slice(None, -2, 2)).assign_coords(magnitude=pmf_mags) - ccmfd.isel(
        magnitude=slice(2, None, 2)
    ).assign_coords(magnitude=pmf_mags)

    return pmf


def get_ccmfd_as_function_b_zeta(b, zeta, mags, mmax, mmin=None):
    """
    Calculate the complimentary cumulative magnitude frequency distribution as function of b and zeta

    Parameters
    ----------
    b: xarray.DataArray
        Slope parameter of the Gutenberg Richter magnitude frequency distribution
    zeta: xarray.DataArray
        Taper parameter of the Gutenberg Richter magnitude frequency distribution
    mags: xarray.DataArray
        Array containing the magnitude values on which to the distribution
    mmax: xarray.DataArray
        Array containing the values of the maximum magnitude
    mmin: xarray.DataArray
        (Optional). Array containing values of the minimum magnitude considered. By default the minimum value of input
        mags is used.

    Returns
    -------
    ccmfd_b_zeta: xarray.DataArray
        complimentary cumulative magnitude frequency distribution as function of b and zeta
    """
    if mmin is None:
        mmin = mags.min()

    a = 10.0 ** (-1.0 * b * (mmax - mmin)) * np.exp(-zeta * (10 ** (1.5 * (mmax - mmin)) - 1))
    v = 10 ** (-1.0 * b * (mags - mmin)) * np.exp(-zeta * (10 ** (1.5 * (mags - mmin)) - 1))
    ccfmd_b_zeta = ((v - a) / (1 - a)).clip(min=0.0)

    return ccfmd_b_zeta


def get_joint_pmf_b_zeta(mag_posterior, cov, b_discr=0.05):
    """
    Calculate the probability mass function of (tapered) Gutenberg Richter parameter b and zeta as a function of cov

    Parameters
    ----------
    mag_posterior: xarray.DataArray
        Posterior distribution of magnitude model
    cov: xarray.DataArray
        Covariate data array
    b_discr: float
        Bin size of b-value

    Returns
    -------
    b_zeta_pmf: xarray.DataArray
        Probability mass function of (tapered) Gutenberg Richter parameter b and zeta as a function of cov

    """
    # Get b-value and zeta, depending on functional form of magnitude model
    b, zeta = get_b_zeta(mag_posterior, cov)

    if zeta.min() == zeta.max():
        b_zeta_pmf = go.aggregate_to_grid(
            samples=b,
            marginalize_dims=mag_posterior.dims,
            target_step=b_discr,
            weights=mag_posterior,
            target="b"
        )
        b_zeta_pmf = b_zeta_pmf.expand_dims({"zeta": np.atleast_1d(zeta.min())})

    else:
        # If we ever need this, we'll need to implement a 2d histogram. Rest of codebase should already be
        # able to deal with the output
        raise UserWarning("No implementation for tapered magnitude distribution available")

    return b_zeta_pmf


def get_b_zeta(mag_posterior, cov):
    """
    Calculate the b and zeta values, depending on the functional form of the magnitude model

    Parameters
    ----------
    mag_posterior: xarray.DataArray
        Posterior distribution of magnitude model
    cov: xarray.DataArray
        Covariate data array

    Returns
    -------
    b: xarray.DataArray
        b-values
    zeta: xarray.DataArray
        zeta-values

    """
    if mag_posterior.name == "linear_stress":
        b0, b_slope = mag_posterior.b0, mag_posterior.b_slope
        res = mag_covariate_linear(cov, b0, b_slope)
        b, zeta = res["b"], res["zeta"]
    elif mag_posterior.name == "split_thickness":
        b_low, b_high, split_location = mag_posterior.b_low, mag_posterior.b_high, mag_posterior.split_location
        res = mag_covariate_split(cov, b_low, b_high, split_location, scale=mag_posterior.attrs['scale'])
        b, zeta = res["b"], res["zeta"]
    else:
        raise UserWarning(f"Functional form of magnitude model {mag_posterior.mag_model} not supported")

    return b, zeta


def discretize_variable(variable, resolution=101, start=None):
    """
    Create a 1D discretized version of 'variable'. Useful to precalculate expensive functions on a limited number
    of points, and then interpolate the result

    Parameters
    ----------
    variable: xarray.DataArray
        Array containing values to be discretized
    resolution: float
        Number of points in the discretized distribution
    mids: bool a second array with 'midpoints', assuming the primary array contains edges
        If True, also return midpoints
    start: float
        A custom value to start the discretization on. If 'None', distribution will start at variable.min

    Returns
    -------
    discr_var: xarray.DataArray
        The discretized distribution
    discr_var_mid: xarray.DataArray (optional if mids==True)
        The midpoints of the discretized distribution
    """

    if start is None:
        start = variable.min()

    vals = go.make_xarray_based(variable.name, np.linspace(start, variable.max().item(), num=resolution))
    return vals
