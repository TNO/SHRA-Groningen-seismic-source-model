import sys
import time
import datetime
import numpy as np
import pandas as pd
from models.rates import bo_2017_forecast
from models.stresses import bo_2017_stress
from tools.catalogue_tools import filter_attr_str2list
from chaintools.chaintools import tools_configuration as cfg
from chaintools.chaintools import tools_xarray as xf
from chaintools.chaintools import tools_grid as go


def main(config_path):
    """
    Create a forecast/hindcast based on the Bayesian calibration of the source model parameters. Store results to file

    Parameters
    ----------
    config_path : str
        The filepath to the configuration file (example included in the repository)
    legacy_format : bool
        Temporary flag for saving results in legacy format
    """

    # Load the configuration file
    module_name = "forecast_ssm"
    config = cfg.configure(config_path, module_name)

    # To make these files, run ssm_input_parser.py
    faults = xf.open("fault_data", config)
    pressure = xf.open("pressure_data", config)
    compaction_coef = xf.open("compressibility_data", config)

    # Load calibration result
    calib_group = config["data_sources"]["calibration_data"]["group"]
    config["data_sources"]["calibration_data"]["group"] = f"{calib_group}/activity_rate_model"
    ar_posterior = xf.open("calibration_data", config)
    config["data_sources"]["calibration_data"]["group"] = f"{calib_group}/magnitude_model"
    mag_posterior = xf.open("calibration_data", config)
    config["data_sources"]["calibration_data"]["group"] = f"{calib_group}/dsm_pmf"
    stress_posterior = xf.open("calibration_data", config)

    stress = bo_2017_stress(
        pressure,
        compaction_coef,
        faults,
        ar_posterior.hs_exp,
        ar_posterior.sigma,
        ar_posterior.rmax,
    )

    # Describe for which period we want a rate forecast (no magnitudes) and for which period we want a full forecast
    calibration_daterange = pd.to_datetime(
        filter_attr_str2list(ar_posterior.attrs["calibration_date_filter"]), format=r"%Y%m%d"
    )
    rate_bool = stress.time >= calibration_daterange[0]
    full_bool = np.logical_and(
        stress.time >= pd.to_datetime(min(config["forecast_epochs"]), format=r"%Y"),
        stress.time <= pd.to_datetime(max(config["forecast_epochs"]) + 1, format=r"%Y"),
    )
    mmax = go.make_xarray_based("mmax", [4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
    mags = go.make_xarray_based("magnitude", np.linspace(1.45, 6.55, 103))

    event_rate_forecast, full_forecast, nr_event_pmf = bo_2017_forecast(
        ar_posterior,
        mag_posterior,
        stress_posterior,
        stress,
        mags,
        mmax,
        rate_bool,
        full_bool,
    )

    # Store results
    forecast_group = config["data_sinks"]["forecast_data"]["group"]
    xf.store(event_rate_forecast, "forecast_data", config, group="event_rate_forecast", mode="a")
    xf.store(full_forecast, "forecast_data", config, group="forecast", mode="a")
    xf.store(nr_event_pmf, "forecast_data", config, group="nr_event_pmf", mode="a")


if __name__ == "__main__":
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    conf_path = sys.argv[1] if sys.argv[1:] else "example_configs/forecast_config.json"
    main(conf_path)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1-time0)))} (hh:mm:ss)")
