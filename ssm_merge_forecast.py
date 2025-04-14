import sys
import time
import datetime
import xarray as xr
from calibrate_ssm import main as main_calibration
from forecast_ssm import main as main_forecast
from chaintools.chaintools import tools_configuration as cfg
from chaintools.chaintools import tools_xarray as xf

def main(arguments):
    """
    This module merges a number of forecast outputs into a single file.

    :param arguments:
    :return:
    """
    config = cfg.configure(arguments)

    # allocate
    eventrate_list = []
    forecast_list = []
    event_pmf_list = []

    # open forecast output files and remove names so that they can be merged easily w. combine_by_coords
    for file_name in config['data_sources'].keys():
        eventrate_list.append(xf.open(file_name, config, group='event_rate_forecast').rename(None))
        forecast_list.append(xf.open(file_name, config, group='forecast').rename(None))
        event_pmf_list.append(xf.open(file_name, config, group='nr_event_pmf').rename(None))

    # merge forecast output files
    xf.store(
        xr.combine_by_coords(eventrate_list).rename('event_rate_forecast'),
        "forecast_data", config, group="event_rate_forecast", mode="w"
    )
    xf.store(
        xr.combine_by_coords(forecast_list).rename('full_forecast'),
        "forecast_data", config, group="forecast", mode="a"
    )
    xf.store(
        xr.combine_by_coords(event_pmf_list).rename('nr_event_pmf'),
        "forecast_data", config, group="nr_event_pmf", mode="a"
    )


if __name__ == '__main__':
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    args = sys.argv[1:] if sys.argv[1:] else \
        ['example_configs/seismic_source_model_config.yml', '--task=ssm_merge_forecast']
    main(args)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1 - time0)))} (hh:mm:ss)")



   