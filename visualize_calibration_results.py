import os
import matplotlib.pyplot as plt
import xarray as xr
from tools.visualization_tools import plot_grid_search


def main(path_to_calibration_file: str, outpath: str):
    """
    Figures for the posterior distribution of the activity rate model and the posterior distribution of the
    magnitude model.

    Parameters
    ----------
    path_to_calibration_file : str
        path and filename of .h5 file containing calibration results
    outpath : str
        path to the output folder where the images are saved
    Returns
    -------

    """

    activity_rate_posterior_probabilities = xr.load_dataarray(path_to_calibration_file,
                                                           group="calibration/activity_rate_model", engine="h5netcdf")
    magnitude_posterior_probabilities = xr.load_dataarray(path_to_calibration_file,
                                                       group="calibration/magnitude_model", engine="h5netcdf")

    # posterior activity rate model
    plot_grid_search(activity_rate_posterior_probabilities)
    plt.savefig(os.path.join(outpath, "activity_model_posterior_grid.png"), dpi=300)
    plt.close()

    # posterior magnitude model
    plot_grid_search(magnitude_posterior_probabilities)
    plt.savefig(os.path.join(outpath, "magnitude_model_posterior_grid.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    out_path = ''
    calibration_results_filepath = 'tests/res/ssm_calibration_test.h5'

    main(calibration_results_filepath, out_path)
