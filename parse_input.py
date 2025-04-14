import sys
import time
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import scipy.interpolate as interp
from tools.fault_tools import get_faults
from tools.catalogue_tools import add_interevent_stats_to_catalogue, get_catalogue, filter_catalogue
from tools.polygon_tools import get_polygon_gdf_from_file
from chaintools.chaintools import tools_configuration as cfg
from chaintools.chaintools import tools_xarray as xf


def main(args: list):
    """
    Parse all the input files to be used in the seismological source model into 'ready to use' files

    Parameters
    ----------
    args : list of str
        First entry has to be the filepath to the configuration file (example included in the repository)
        Optionally contains '--task=TASK_NAME'
    """

    # Load the configuration file
    config = cfg.configure(args)

    # Parse and save the earthquake catalogue
    raw_catalogue = get_catalogue(
        config['data_sources']['raw_eq_file']
    )

    polygon = get_polygon_gdf_from_file(
        xf.construct_path(config['data_sources']['calibration_polygon_file']['path'])
    )
    filters = {
        'magnitude': [config.get('mmin_catalogue', 1.45) , config.get('mmax_catalogue', None)],
        'date': config['calibration_date_range'],
        'location': polygon,
    }
    catalogue = filter_catalogue(raw_catalogue, filters)
    catalogue = add_interevent_stats_to_catalogue(catalogue)
    xf.store(catalogue, 'eq_catalogue', config)

    # Parse the pressure grid, reservoir depth grid, and compressibility grid
    pressure = get_grid_data(
        xf.construct_path(config['data_sources']['raw_press_file']['path'])
    )
    res_depth = get_grid_data(
        xf.construct_path(config['data_sources']['raw_depth_file']['path'])
    )
    res_thickness = get_grid_data(
        xf.construct_path(config['data_sources']['raw_thickness_file']['path'])
    )
    compr = get_grid_data(
        xf.construct_path(config['data_sources']['raw_compr_file']['path'])
    ).expand_dims(dim={'cm_grid_version': ['cm_NAM2018']})

    # (optional) compressibility grid for rticm model
    if 'raw_compr_rticm_file' in config['data_sources'].keys():
        compr_rticm = get_grid_data(
            xf.construct_path(config['data_sources']['raw_compr_rticm_file']['path'])
        ).expand_dims(dim={'cm_grid_version': ['cm_NAM2021']})
        compr_rticm = compr_rticm.interp_like(
            compr.isel(cm_grid_version=0),
            kwargs={"bounds_error": False, "fill_value": 0.0}
        )
        compr = xr.concat([compr, compr_rticm], dim='cm_grid_version')

    # Ensure consistent spatial grid
    spatial_base_grid = config.get('spatial_base_grid', 'pressure')  # 'pressure' (default) or 'compressibility'
    if spatial_base_grid == 'pressure':
        compr = compr.interp_like(
            pressure.isel(time=0), kwargs={"bounds_error": False, "fill_value": 0.0}
        )
        res_depth = res_depth.interp_like(
            pressure.isel(time=0), kwargs={"bounds_error": False, "fill_value": 0.0}
        )
        res_thickness = res_thickness.interp_like(
            pressure.isel(time=0), kwargs={"bounds_error": False, "fill_value": 0.0}
        )
    elif spatial_base_grid == 'compressibility':
        pressure = pressure.interp_like(
            compr.isel(cm_grid_version=0), kwargs={"bounds_error": False, "fill_value": 0.0}
        )
        res_depth = res_depth.interp_like(
            compr.isel(cm_grid_version=0), kwargs={"bounds_error": False, "fill_value": 0.0}
        )
        res_thickness = res_thickness.interp_like(
            compr.isel(cm_grid_version=0), kwargs={"bounds_error": False, "fill_value": 0.0}
        )
    else:
        raise UserWarning(f'Base grid "{spatial_base_grid}" not supported. Choose "pressure" or "compressibility"')

    # Save the pressure grid and the compressibility grid
    xf.store(pressure, 'pressure_data', config)
    xf.store(compr, 'compressibility_data', config)
    xf.store(res_depth, 'reservoir_depth_data', config)
    res_thickness = res_thickness.where(res_thickness > 1, np.nan)
    xf.store(res_thickness, 'reservoir_thickness_data', config)

    # Parse and save the faults file
    faults = get_faults(
        xf.construct_path(config['data_sources']['raw_fault_file']['path'])
    )
    xf.store(faults, 'fault_data', config)

    print("All files parsed")


def get_grid_data(data_path, method="linear", dxdy=100):
    """
    Reads and parses spatial -temporal data from a raw data file

    Parameters
    ----------
    data_path : str
        Path to the raw data file
    method : str, optional
        Method to use for interpolation, by default 'linear'
    dxdy : int, optional
        Grid spacing, by default 100

    Returns
    -------
    out_grid : xarray.DataArray
        Grid data
    """
    raw_data = pd.read_csv(data_path)

    x_key = "X" if "X" in raw_data.columns else "x"
    y_key = "Y" if "Y" in raw_data.columns else "y"
    prop_keys = [str(a) for a in raw_data.columns[2:]]

    # Determine whether data is already on a grid. If so, use original grid spacing, else use dxdy
    x_u, y_u = np.sort(np.unique(raw_data[x_key])), np.sort(np.unique(raw_data[y_key]))
    x_diff, y_diff = np.diff(x_u), np.diff(y_u)
    xmin, xmax, ymin, ymax = x_u.min(), x_u.max(), y_u.min(), y_u.max()
    xdmin, xdmax, ydmin, ydmax = x_diff.min(), x_diff.max(), y_diff.min(), y_diff.max()
    if np.allclose(xdmin, xdmax) and np.allclose(ydmin, ydmax) and np.allclose(xdmin, ydmax):
        dxdy = xdmin

    # Create grid definition to put data onto
    xlin = np.arange(xmin, xmax + dxdy, dxdy)
    ylin = np.arange(ymin, ymax + dxdy, dxdy)
    points = np.asarray([raw_data[x_key].tolist(), raw_data[y_key].tolist()]).T
    xgrid, ygrid = np.meshgrid(xlin, ylin)
    coords = {"y": ylin, "x": xlin}

    if len(prop_keys) == 1:
        prop_name = prop_keys[0]
        grid = interp.griddata(points, raw_data[prop_name], (xgrid, ygrid), method=method)
    else:
        prop_name = "pressure"  # Little hacky to just assume, but we don't use other dynamic grids
        timesteps = [pd.to_datetime(ts) for ts in sorted(p[1:].replace("_", "") for p in prop_keys)]
        coords = {"time": timesteps} | coords
        grid = np.zeros(shape=(len(timesteps), len(ylin), len(xlin)))
        for i, prop in enumerate(prop_keys):
            grid[i, :, :] = interp.griddata(points, raw_data[prop], (xgrid, ygrid), method=method)

    out_grid = xr.DataArray(grid, coords=coords, name=prop_name).rio.write_crs(28992, inplace=True)
    # This seems pointless, but allows saving and loading as DataArray (which is otherwise prevented by rio?)
    vals = np.copy(out_grid.values)
    out_grid = out_grid.interp_like(out_grid)
    out_grid.values = vals
    return out_grid


if __name__ == "__main__":
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    args = sys.argv[1:] if sys.argv[1:] else ["example_configs/parser_config.yml", "--task=parse_input_ssm"]
    main(args)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1 - time0)))} (hh:mm:ss)")
