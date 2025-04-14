import os
import sys
import time
import datetime
import matplotlib.pyplot as plt
from chaintools.chaintools import tools_configuration as cfg
import xarray as xr
import numpy as np
from chaintools.chaintools import tools_xarray as xf
from tools.visualization_tools import plot_and_save_annual_event_density_maps, plot_and_save_annual_magnitude_model, \
    plot_and_save_fieldwide_event_counts, write_field_csv

def main(args: list):
    """
    Create figures from the seismic source model forecast results.
    - Field-wide observed and simulated event counts versus time
    - Event density maps of the field for each forecasted time interval
    - Magnitude-frequency plots for each forecasted time interval

    Parameters
    ----------
    args : list of str
        First entry has to be the filepath to the configuration file (example included in the repository)
        Optionally contains '--task=TASK_NAME'
    """

    # Load the configuration file
    config = cfg.configure(args)

    # Load forecast result
    fc_group = config["data_sources"]["forecast_data"]["group"]
    config["data_sources"]["forecast_data"]["group"] = f"{fc_group}/forecast"
    full_forecast = xf.open("forecast_data", config)
    config["data_sources"]["forecast_data"]["group"] = f"{fc_group}/event_rate_forecast"
    event_rate_forecast = xf.open("forecast_data", config)
    config["data_sources"]["forecast_data"]["group"] = f"{fc_group}/nr_event_pmf"
    event_uncertainty = xf.open("forecast_data", config)
    eq_catalogue = xf.open("eq_catalogue", config)
    alt_eq_catalogue = config["data_sources"].get("alt_eq_catalogue", False)
    if alt_eq_catalogue is not False:
        alt_eq_catalogue = xf.open("alt_eq_catalogue", config)

    out_path = xf.construct_path(config["out_path"])

    # get mean over logic tree branches, if provided. NOTE: mmax branch is ignored.
    if 'logic_tree' in config['data_sources'].keys():
        logic_tree = xf.open('logic_tree', config)
        logic_tree, model_dims = xf.prepare_weights(logic_tree, event_rate_forecast)

        lt_weights = (
            xr.dot(*[logic_tree[var] for var in logic_tree.data_vars])
            .expand_dims({'weight_version': ['mean']})
        )
    else:
        lt_weights = None
        # identify model dimensions
        model_dims = [d for d in event_rate_forecast.dims if d not in ['x', 'y', 'time']]

    # set weights for individual model combinations
    weights = (
        xr.dot(*[xr.ones_like(event_rate_forecast[d], dtype=float) for d in model_dims])
        .stack({'_model_':model_dims})
    )
    weights = weights.expand_dims({'weight_version' : weights._model_.values})
    weights.data = np.diag(np.ones(len(weights._model_)))
    weights = weights.unstack('_model_')

    # merge with logic tree weights
    if lt_weights is not None:
        weights = xr.concat([weights, lt_weights], dim='weight_version')

    for weight in weights.weight_version:

        er = event_rate_forecast.weighted(weights.sel(weight_version=weight)).sum(model_dims)
        eu = event_uncertainty.weighted(weights.sel(weight_version=weight)).sum(model_dims)
        ff = full_forecast.weighted(weights.sel(weight_version=weight)).sum(model_dims)

        model_name = weight.values[()]
        if isinstance(model_name, tuple):
            clean_model_name = '-'.join(weight.values[()])
        else:
            clean_model_name = model_name

        if alt_eq_catalogue is not False:
            plot_and_save_fieldwide_event_counts(
                er,
                eu,
                eq_catalogue,
                alternative_catalogue_path=alt_eq_catalogue,
                fig_path=os.path.join(out_path, f"nr_events_all_{clean_model_name}.png")
            )

        else:
            write_field_csv(ff, lt_weights, csv_path=os.path.join(out_path, f"nr_events_{clean_model_name}.csv"))

            plot_and_save_fieldwide_event_counts(
                er,
                eu,
                eq_catalogue,
                fig_path=os.path.join(out_path, f"nr_events_{clean_model_name}.png"))
            plot_and_save_annual_event_density_maps(
                ff,
                fig_path=os.path.join(out_path, f"annual_event_density_{clean_model_name}.png"))
            plot_and_save_annual_magnitude_model(
                ff,
                basedir=os.path.join(out_path, f'annual_fmd_{clean_model_name}'))


if __name__ == "__main__":
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    args = sys.argv[1:] if sys.argv[1:] else ["example_configs/seismic_source_model_config.yml"]
    main(args)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1 - time0)))} (hh:mm:ss)")