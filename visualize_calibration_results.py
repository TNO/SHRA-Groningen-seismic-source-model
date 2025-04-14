import sys
import time
import datetime
from tools.visualization_tools import plot_and_save_grid_evaluation
from chaintools.chaintools import tools_configuration as cfg
from chaintools.chaintools import tools_xarray as xf
from pathlib import Path

def main(args: list):
    """
    Figures for the posterior distribution of the activity rate model and the posterior distribution of the
    magnitude model.

    Parameters
    ----------
    args : list of str
        First entry has to be the filepath to the configuration file (example included in the repository)
        Optionally contains '--task=TASK_NAME'
    """

    # Load the configuration file
    config = cfg.configure(args)

    # Load calibration result
    calib_group = config["data_sources"]["calibration_data"]["group"]
    config["data_sources"]["calibration_data"]["group"] = f"{calib_group}/activity_rate_model"
    ar_posterior = xf.open("calibration_data", config)
    config["data_sources"]["calibration_data"]["group"] = f"{calib_group}/magnitude_model"
    mag_posterior = xf.open("calibration_data", config)
    out_path = xf.construct_path(config["out_path"])

    # Plot posterior activity rate model and posterior magnitude model
    for mag_model in list(mag_posterior.data_vars.keys()):
        fig_path = Path().joinpath(
            *[out_path, "magnitude_model_posterior_grid_{}_{}.png".format(mag_model, ar_posterior.branch_rate)]
        )
        plot_and_save_grid_evaluation(mag_posterior[mag_model], fig_path)
        fig_path = Path().joinpath(
            *[out_path, "activity_model_posterior_grid_{}_{}.png".format(mag_model, ar_posterior.branch_rate)]
        )
        plot_and_save_grid_evaluation(ar_posterior[mag_model], fig_path)


if __name__ == "__main__":
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    args = sys.argv[1:] if sys.argv[1:] else ["example_configs/seismic_source_model_config.yml"]
    main(args)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1 - time0)))} (hh:mm:ss)")
