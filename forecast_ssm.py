import sys
import time
import datetime
import numpy as np
import pandas as pd
from models.rates import bo_2017_forecast
from models.stresses import bo_2017_stress
from tools.catalogue_tools import filter_attr_str2list
from tools.polygon_tools import get_polygon_gdf_from_points
from chaintools.chaintools import tools_configuration as cfg
from chaintools.chaintools import tools_xarray as xf


def main(arguments: list):
    """
    Create a forecast/hindcast based on the Bayesian calibration of the source model parameters. Store results to file

    Parameters
    ----------
    arguments : list of str
        First entry has to be the filepath to the configuration file (example included in the repository)
        Optionally contains '--task=TASK_NAME'
    """

    # Load the configuration file
    config = cfg.configure(arguments)

    # To make these files, run ssm_input_parser.py
    faults = xf.open("fault_data", config)
    pressure = xf.open("pressure_data", config)
    compaction_coef = xf.open("compressibility_data", config)
    reservoir_depth = xf.open("reservoir_depth_data", config)
    reservoir_thickness = xf.open("reservoir_thickness_data", config)
    catalogue = xf.open("eq_catalogue", config)

    # Load calibration result
    calib_group = config["data_sources"]["calibration_data"]["group"]
    config["data_sources"]["calibration_data"]["group"] = f"{calib_group}/activity_rate_model"
    ar_posterior = xf.open("calibration_data", config)
    config["data_sources"]["calibration_data"]["group"] = f"{calib_group}/magnitude_model"
    mag_posterior = xf.open("calibration_data", config)
    config["data_sources"]["calibration_data"]["group"] = f"{calib_group}/dsm_pmf"
    stress_posterior = xf.open("calibration_data", config)
    config["data_sources"]["calibration_data"]["group"] = f"{calib_group}/logic_tree_info"

    polygon = get_polygon_gdf_from_points(catalogue.location_filter_x, catalogue.location_filter_y)

    params = xf.prepare_ds(config)
    # add param space from calibration
    params = params.expand_dims({d: ar_posterior[d].values for d in ar_posterior.coords if ar_posterior[d].shape})

    stress = bo_2017_stress(
        pressure,
        compaction_coef,
        reservoir_depth,
        faults,
        params,
        out_name=ar_posterior.attrs['branch_rate']
    )
    stress['reservoir_thickness'] = reservoir_thickness

    # Describe for which period we want a rate forecast (no magnitudes) and for which period we want a full forecast
    calibration_daterange = pd.to_datetime(
        filter_attr_str2list(ar_posterior.attrs["calibration_date_filter"]), format=r"%Y%m%d"
    )
    rate_bool = stress.time >= calibration_daterange[0]
    full_bool = np.logical_and(
        stress.time >= pd.to_datetime(min(config["forecast_epochs"]), format=r"%Y"),
        stress.time <= pd.to_datetime(max(config["forecast_epochs"]) + 1, format=r"%Y"),
    )

    event_rate_forecast, full_forecast, nr_event_pmf = bo_2017_forecast(
        ar_posterior,
        mag_posterior,
        stress_posterior,
        stress,
        params['magnitude'],
        params['branch_mmax'],
        rate_bool,
        full_bool,
        polygon
    )

    # Add names so conversion to xarray datasets can be done safely
    rate_model = ar_posterior.attrs['branch_rate']
    event_rate_forecast = event_rate_forecast.expand_dims({'branch_rate': [rate_model]}).rename('event_rate_forecast')
    full_forecast = full_forecast.expand_dims({'branch_rate': [rate_model]}).rename('full_forecast')
    nr_event_pmf = nr_event_pmf.expand_dims({'branch_rate': [rate_model]}).rename('nr_event_pmf')

    # Store results
    xf.store(event_rate_forecast, "forecast_data", config, group="event_rate_forecast", mode="w")
    xf.store(full_forecast, "forecast_data", config, group="forecast", mode="a")
    xf.store(nr_event_pmf, "forecast_data", config, group="nr_event_pmf", mode="a")


if __name__ == "__main__":
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    args = sys.argv[1:] if sys.argv[1:] else ["example_configs/seismic_source_model_config.yml"]
    main(args)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1 - time0)))} (hh:mm:ss)")
