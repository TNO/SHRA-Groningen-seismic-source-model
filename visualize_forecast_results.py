import os
import xarray as xr
import matplotlib.pyplot as plt
from tools.visualization_tools import plot_annual_event_density_maps, plot_and_save_annual_magnitude_model, \
    plot_fieldwide_event_counts


def main(path_to_forecast_file: str, path_to_eq_catalogue: str, outpath: str):
    """
    Create figures from the seismic source model forecast results.
    - Field-wide observed and simulated event counts versus time
    - Event density maps of the field for each forecasted time interval
    - Magnitude-frequency plots for each forecasted time interval
    :param path_to_forecast_file: path to forecast result, in .h5 file format
    :param path_to_eq_catalogue: path to observed earthquake catalogue, in .h5 file format (given as output from
                                 parse_input.py)
    :param outpath: path to directory where figures will be stored
    """

    total_forecast = xr.load_dataarray(path_to_forecast_file, group="forecast/forecast", engine="h5netcdf")
    event_rate_forecast = xr.load_dataarray(path_to_forecast_file, group="forecast/event_rate_forecast",
                                            engine="h5netcdf")
    total_event_count_uncertainty = xr.load_dataarray(path_to_forecast_file, group="forecast/nr_event_pmf",
                                                      engine="h5netcdf")

    plot_fieldwide_event_counts(
        event_rate_forecast,
        total_event_count_uncertainty,
        path_to_eq_catalogue,
        plot_incomplete_intervals=True
    )
    plt.savefig(os.path.join(outpath, "nr_events.png"), dpi=300)
    plt.close()

    plot_annual_event_density_maps(total_forecast)
    plt.savefig(os.path.join(outpath, "annual_event_density.png"), dpi=300)
    plt.close()

    plot_and_save_annual_magnitude_model(total_forecast, outpath)


if __name__ == "__main__":
    out_path = ''
    forecast_results_path = 'tests/res/ssm_forecast_test.h5'
    eq_catalogue_path = 'tests/res/eq_cat.h5'

    main(forecast_results_path, eq_catalogue_path, out_path)
