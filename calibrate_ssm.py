import sys
import time
import datetime
import xarray as xr
from tools.bayes_tools import (
    poisson_log_likelihood,
    magnitude_log_likelihood,
    get_posterior_probability,
    get_combined_posterior,
)
from models.magnitudes import mag_covariate_linear, mag_covariate_split
from models.rates import bo_2017_ll_terms
from models.stresses import bo_2017_stress
from models.etas import get_etas_rates
from tools.catalogue_tools import covariate_at_event
from chaintools.chaintools import tools_configuration as cfg
from chaintools.chaintools import tools_xarray as xf


def main(args: list):
    """
    Perform the Bayesian calibration of the source model parameters. Store results to file

    Parameters
    ----------
    args : list of str
        First entry has to be the filepath to the configuration file (example included in the repository)
        Optionally contains '--task=TASK_NAME'
    """

    config = cfg.configure(args)

    # To make these files, run ssm_input_parser.py
    catalogue = xf.open("eq_catalogue", config)
    faults = xf.open("fault_data", config)
    pressure = xf.open("pressure_data", config)
    compaction_coef = xf.open("compressibility_data", config)
    reservoir_depth = xf.open("reservoir_depth_data", config)

    # Define the parameter ranges for which to calibrate
    params = xf.prepare_ds(config)
    reservoir_thickness = xf.open("reservoir_thickness_data", config)

    stress = bo_2017_stress(
        pressure,
        compaction_coef,
        reservoir_depth,
        faults,
        params
    )

    # Calculate the log-likelihood for the activity rate model
    background_res = bo_2017_ll_terms(stress, catalogue, params.theta0, params.theta1)
    etas_res = get_etas_rates(catalogue, etas_k=params.etas_k, etas_a=params.etas_a)
    ar_log_likelihood = poisson_log_likelihood(background_res, etas_res, use_loop=True)

    # Calculate the log-likelihoods of the magnitude models and store in dataset:
    mag_log_likelihood = xr.Dataset()
    # Stress-dependent magnitude model
    stress_at_events = covariate_at_event(stress["dcs"], catalogue)
    fmd_terms = mag_covariate_linear(stress_at_events, params.b0, params.b_slope)
    mag_log_likelihood["linear_stress"] = magnitude_log_likelihood(fmd_terms, catalogue, dm=0.0, use_dask=True)
    # Split-thickness magnitude model
    thickness_at_events = covariate_at_event(reservoir_thickness, catalogue, rate=False)
    fmd_terms = mag_covariate_split(thickness_at_events, params.b_low, params.b_high, params.split_location)
    mag_log_likelihood["split_thickness"] = magnitude_log_likelihood(fmd_terms, catalogue, dm=0.0, use_dask=False)

    # Rearrange activity rate ll arrays for the magnitude-frequency models
    mm_selection = [mf for mf in mag_log_likelihood.data_vars]
    ar_log_likelihood = ar_log_likelihood.expand_dims({'mm': mm_selection}).to_dataset(dim='mm')

    # Get posterior probability as well as combined post_prob of stress model parameters
    with xr.set_options(keep_attrs=True):
        ar_posterior = get_posterior_probability(ar_log_likelihood)
        mag_posterior = get_posterior_probability(mag_log_likelihood)
        keep_dims = [d for d in stress.dims if d not in ['x', 'y', 'time']]
        stress_posterior = get_combined_posterior(
            ar_posterior,
            mag_posterior,
            keep_dims=keep_dims
        )

    # Include calibration and model information in the posterior before saving
    ar_posterior.attrs.update({f"calibration_{key}": catalogue.attrs[key] for key in catalogue.attrs.keys()})
    ar_posterior.attrs.update({key: stress.attrs[key] for key in stress.attrs.keys()})
    mag_posterior.attrs.update({f"calibration_{key}": catalogue.attrs[key] for key in catalogue.attrs.keys()})

    # Save section
    xf.store(ar_posterior, 'calibration_data', config, group="activity_rate_model", mode="w")
    xf.store(mag_posterior, 'calibration_data', config, group="magnitude_model", mode="a")
    xf.store(stress_posterior, 'calibration_data', config, group="dsm_pmf", mode="a")
    xf.store(catalogue, 'calibration_data', config, group="earthquake_data", mode="a")


if __name__ == "__main__":
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    args = sys.argv[1:] if sys.argv[1:] else ["example_configs/seismic_source_model_config.yml", "--task=calibrate_ssm_rticm"]
    main(args)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1 - time0)))} (hh:mm:ss)")
