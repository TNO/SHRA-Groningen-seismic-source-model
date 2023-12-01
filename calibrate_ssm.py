import sys
import time
import datetime
from tools.bayes_tools import (
    poisson_log_likelihood,
    magnitude_log_likelihood,
    get_posterior_probability,
    get_combined_posterior,
)
from models.magnitudes import mag_tanh_bval
from models.rates import bo_2017_ll_terms
from models.stresses import bo_2017_stress
from models.etas import get_etas_rates
from tools.catalogue_tools import covariate_at_event
from chaintools.chaintools import tools_configuration as cfg
from chaintools.chaintools import tools_xarray as xf


def main(config_path):
    """
    Perform the Bayesian calibration of the source model parameters. Store results to file

    Parameters
    ----------
    config_path : str
        The filepath to the configuration file (example included in the repository)
    """

    module_name = "calibrate_ssm"
    config = cfg.configure(config_path, module_name)

    # To make these files, run ssm_input_parser.py
    catalogue = xf.open("eq_catalogue", config)
    faults = xf.open("fault_data", config)
    pressure = xf.open("pressure_data", config)
    compaction_coef = xf.open("compressibility_data", config)

    # Define the parameter ranges for which to calibrate
    params = xf.prepare_ds(config)

    # Calculate the stress
    stress = bo_2017_stress(pressure, compaction_coef, faults, params.hs_exp, params.sigma, params.rmax)

    # Calculate the log-likelhood for the activity rate model
    background_res = bo_2017_ll_terms(stress, catalogue, params.theta0, params.theta1)
    etas_res = get_etas_rates(catalogue, params.etas_k, params.etas_a)
    ar_log_likelihood = poisson_log_likelihood(background_res, etas_res, use_loop=True)

    # Calculate the log-likelhood for the magnitude model
    stress_at_events = covariate_at_event(stress["dcs"], catalogue)
    fmd_terms = mag_tanh_bval(stress_at_events, params.b_theta0, params.b_theta1, params.b_theta2)
    mag_log_likelihood = magnitude_log_likelihood(fmd_terms, catalogue, dm=0.1, use_dask=True)

    # Get posterior probability as well as combined post_prob of stress model parameters
    ar_posterior = get_posterior_probability(ar_log_likelihood)
    mag_posterior = get_posterior_probability(mag_log_likelihood)

    # Include calibration and model information in the posterior before saving
    ar_posterior.attrs.update({f"calibration_{key}": catalogue.attrs[key] for key in catalogue.attrs.keys()})
    mag_posterior.attrs.update({f"calibration_{key}": catalogue.attrs[key] for key in catalogue.attrs.keys()})
    ar_posterior.attrs.update({"ar_model": "bo_2017"})
    mag_posterior.attrs.update({"mag_model": "stress_tanh"})

    # Get the posterior probability for the stress model, based on both the activity rate and magnitude models
    stress_posterior = get_combined_posterior(ar_posterior, mag_posterior)

    # Save section
    calib_group = config["data_sinks"]["calibration_data"]["group"]
    xf.store(ar_posterior, 'calibration_data', config, group=f"{calib_group}/activity_rate_model", mode="w")
    xf.store(mag_posterior, 'calibration_data', config, group=f"{calib_group}/magnitude_model", mode="a")
    xf.store(stress_posterior, 'calibration_data', config, group=f"{calib_group}/dsm_pmf", mode="a")
    xf.store(catalogue, 'calibration_data', config, group=f"{calib_group}/earthquake_data", mode="a")


if __name__ == "__main__":
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    conf_path = sys.argv[1] if sys.argv[1:] else "example_configs/calibration_config.json"
    main(conf_path)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1 - time0)))} (hh:mm:ss)")
