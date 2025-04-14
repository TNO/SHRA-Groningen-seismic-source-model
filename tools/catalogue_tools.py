"""
This module contains functions to read and parse earthquake catalogues sourced from the KNMI.
"""

from functools import lru_cache
import io
import datetime
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import rioxarray  # pylint: disable=unused-import
import chaintools.chaintools.tools_xarray as xf


def add_interevent_stats_to_catalogue(catalogue):
    """
    Add inter-event statistics to the catalogue. These are needed for calibration of ETAS parameters

    Parameters
    ----------
    catalogue : xarray.Dataset
        The earthquake catalogue

    Returns
    -------
     catalogue : xarray.Dataset
        The earthquake catalogue with inter-event statistics added
    """

    nr_eq = len(catalogue.x)
    int_e_dist, int_e_time, diff_mag = np.zeros([nr_eq, nr_eq]), np.zeros([nr_eq, nr_eq]), np.zeros([nr_eq, nr_eq])

    eq_x, eq_y, eq_mag, eq_time = (
        catalogue.x.values,
        catalogue.y.values,
        catalogue.magnitude.values,
        catalogue.date_time.values,
    )

    for i, (first_x, first_y, first_time) in enumerate(zip(eq_x, eq_y, eq_time)):
        for j in range(i):
            second_x, second_y, second_mag, second_time = eq_x[j], eq_y[j], eq_mag[j], eq_time[j]
            int_e_dist[i, j] = (first_x - second_x) ** 2 + (first_y - second_y) ** 2
            int_e_time[i, j] = ((first_time - second_time) / 1e9).astype(
                float
            ) / 31557600  # In Julian years (365.25 days)
            diff_mag[i, j] = second_mag - catalogue.minmag

    dim_names = [list(catalogue.sizes.keys())[0], "dim_1"]
    vals = [catalogue[dim_names[0]], np.copy(catalogue[dim_names[0]])]
    coords = dict(zip(dim_names, vals))

    for stat, name in zip([int_e_dist, int_e_time, diff_mag], ["r2", "int_e_time", "diff_mag"]):
        data_array = xr.DataArray(stat, coords=coords)
        catalogue = catalogue.assign({name: data_array})

    return catalogue


def _parse_knmi_csv(eq_df):
    """
    Parse the KNMI dataframe file into the internal format

    Parameters
    ----------
    eq_df : pandas.Dataframe
        The dataframe containing the data, raw from file/http read

    Returns
    -------
    eq_gdf : geopandas.Dataframe
        The dataframe containing the data, formatted for use in the module
    """
    date = eq_df.YYMMDD.astype(str)

    # old or new TIME format
    if isinstance(eq_df.TIME[0], float):
        time = eq_df.TIME.astype(int).astype(str).str.zfill(6)
        eq_df["date_time"] = pd.to_datetime(date + time, format=r"%Y%m%d%H%M%S")
    else:
        time = eq_df.TIME.astype(str)
        eq_df["date_time"] = pd.to_datetime(date + time, format=r"%Y%m%d%H:%M:%S")

    eq_df.rename(
        columns={"LAT": "lat", "LON": "lon", "DEPTH": "depth", "MAG": "magnitude", "LOCATION": "community"},
        inplace=True,
    )
    eq_df["timestamp"] = xr.DataArray(
        [pd.to_datetime(t).timestamp() for t in eq_df["date_time"].values], dims="date_time"
    )
    rd_locs = gpd.points_from_xy(eq_df["lon"], eq_df["lat"], crs=4326).to_crs(28992)
    eq_df["x"] = rd_locs.x
    eq_df["y"] = rd_locs.y

    eq_df.drop(columns=["EVALMODE", "YYMMDD", "TIME"], inplace=True)
    eq_gdf = gpd.GeoDataFrame(eq_df, geometry=rd_locs)

    return eq_gdf


@lru_cache  # do not bother KNMI with repeated calls
def _get_catalogue_rdsa(**pars):
    """
    Get the KNMI data from the KNMI website (RDSA) and parse into dataframe
    Parameters
    ----------
    pars : dict
        The parameters to pass to the KNMI website

    Returns
    -------
    eq_gdf : geopandas Dataframe
        The dataframe containing the data, formatted for use in the module
    data_source : str
        Description of the source of the data
    """
    def_pars = {
        "minlatitude": 53.0,
        "maxlatitude": 53.7,
        "minlongitude": 6.4,
        "maxlongitude": 7.3,
        "format": "csv",
    }
    params = def_pars | pars
    data_source = "http://rdsa.knmi.nl/fdsnws/event/1/query"
    request = requests.get(data_source, params=params, timeout=60)
    eq_df = pd.read_csv(
        io.StringIO(request.text),
        header=None,
        names=["knmi_id", "date_time", "lat", "lon", "depth", "magnitude", "community"],
    ).rename({"knmi_id": "dim_0"}, axis="columns").set_index("dim_0")
    eq_df["date_time"] = pd.to_datetime(eq_df["date_time"])
    eq_df["timestamp"] = xr.DataArray(
        [pd.to_datetime(t).timestamp() for t in eq_df["date_time"].values], dims="date_time"
    )
    rd_locs = gpd.points_from_xy(eq_df["lon"], eq_df["lat"], crs=4326).to_crs(28992)
    eq_df["x"] = rd_locs.x
    eq_df["y"] = rd_locs.y
    eq_gdf = gpd.GeoDataFrame(eq_df, geometry=rd_locs)

    return eq_gdf, data_source


@lru_cache  # do not bother KNMI with repeated calls
def _get_catalogue_http():
    """
    Get the KNMI data from the KNMI website and parse it into a geodataframe.

    Returns
    -------
    eq_gdf : geopandas.Dataframe
        The dataframe containing the data, formatted for use in the module
    data_source : str
        Description of the source of the data
    """
    data_source = "http://cdn.knmi.nl/knmi/map/page/seismologie/all_induced.csv"
    eq_df = pd.read_csv(data_source)
    eq_gdf = _parse_knmi_csv(eq_df)

    return eq_gdf, data_source


def _get_catalogue_file(path):
    """
    Get the KNMI data from a local file and parse it into a geodataframe.

    Parameters
    ----------
    path : str
        The path to the local file

    Returns
    -------
    eq_gdf : geopandas.Dataframe
        The dataframe containing the data, formatted for use in the module
    data_source : str
        Description of the source of the data
    """
    data_source = path
    eq_df = pd.read_csv(data_source)
    eq_gdf = _parse_knmi_csv(eq_df)

    return eq_gdf, data_source


def get_catalogue(pars: dict):
    """
    Get the KNMI data from any of the supported query types and parse to xarray.
    Use rioxarray to georeference the data according to CF conventions
    https://corteva.github.io/rioxarray/stable/getting_started/crs_management.html

    Parameters
    ----------
    query_type : str
        The type of query to perform. Supported:
            - 'rdsa': use the RDSA earthquake data
            - 'http': get the 'all_induced.csv' directly from the KNMI website
            - 'file': use the local 'all_induced.csv' file (only supported in pars
                      contains 'local_path' pointing to the csv)
    **pars : any
        The keyword-pair value parameters to pass to the query.

    Returns
    -------
    eq_xr : xarray.Dataset
        The dataset containing the data, formatted for use in the module
    """
    access_time = str(datetime.datetime.utcnow())
    local_path = pars.pop("path", None)
    query_type = pars.pop("type", "raw")
    if query_type == "rdsa":
        eq_gdf, data_source = _get_catalogue_rdsa(**pars)
    elif query_type == "http":
        eq_gdf, data_source = _get_catalogue_http()
    elif query_type == "file" or query_type == "raw":
        local_path = xf.construct_path(local_path)
        eq_gdf, data_source = _get_catalogue_file(local_path)
        access_time = 0  # Not relevant for file access, and breaks reuseability
    else:
        raise UserWarning(
            f'query type: {query_type} is unknown. Available query types are: "rdsa", "http", and "file"'
        )

    eq_xr = xr.Dataset(eq_gdf).rio.write_crs(eq_gdf.crs, inplace=True).set_coords(["x", "y", "date_time"])
    eq_xr["geometry"] = eq_xr.geometry.astype(str)  # Fix dtype
    eq_xr = eq_xr.assign_attrs({"data_source": str(data_source), "access_time": access_time, "minmag": float("Nan")})

    # ensure order from old to new events
    eq_xr = eq_xr.sortby(eq_xr.date_time)

    if eq_xr.dim_0.dtype == object or  eq_xr.dim_0.dtype == str:
        eq_xr = eq_xr.assign_coords({'dim_0':range(len(eq_xr.dim_0))})

    return eq_xr


def filter_catalogue(catalogue, filter_dict):
    """
    Filter the catalogue dataframe based on the filter_dict.

    Parameters
    ----------
    catalogue : xarray.Dataset
        The catalogue dataframe to filter.
    filter_dict : dict
        The filter_dict to use. Allowed keys: date, magnitude, location. Other keys are ignored
            date: List of [min, max]. Use of 'None' is allowed
            magnitude: List of [min, max]. Use of 'None' is allowed
            location: geopandas.Dataframe.

    Returns
    -------
    catalogue : xarray.Dataset
        The filtered catalogue dataframe.
    """
    # Allow 'no filter'
    base_filter = np.array([True for _ in catalogue.x])
    filters = [base_filter]
    filter_attrs = {}

    if "date" in filter_dict:
        if filter_dict["date"][0] is not None:
            start_day = pd.to_datetime(filter_dict["date"][0], format=r"%Y%m%d")
            filters.append((catalogue.date_time >= start_day).values)
        if filter_dict["date"][1] is not None:
            # Include the whole final day
            end_day = (
                    pd.to_datetime(filter_dict["date"][1], format=r"%Y%m%d")
                    + pd.DateOffset(days=1)
                    - pd.DateOffset(seconds=1)
            )
            filters.append((catalogue.date_time <= end_day).values)
        filter_attrs["date_filter"] = repr(filter_dict["date"])

    if "magnitude" in filter_dict:
        if filter_dict["magnitude"][0] is not None:
            filters.append((catalogue.magnitude >= filter_dict["magnitude"][0]).values)
            filter_attrs["minmag"] = filter_dict["magnitude"][0]
        if filter_dict["magnitude"][1] is not None:
            filters.append((catalogue.magnitude <= filter_dict["magnitude"][1]).values)
        filter_attrs["magnitude_filter"] = repr(filter_dict["magnitude"])

    if "location" in filter_dict:
        polygon = filter_dict["location"]
        points = gpd.GeoSeries(gpd.points_from_xy(catalogue["x"], catalogue["y"], crs=28992))
        # Keep points inside the polygon
        filters.append(np.array([polygon.contains(p).values[0] for p in points]))
        filter_attrs["location_filter"] = polygon.attrs["field_id"]
        filter_attrs["location_filter_x"] = polygon.unary_union.boundary.coords._coords[:, 0]
        filter_attrs["location_filter_y"] = polygon.unary_union.boundary.coords._coords[:, 1]

    filters = np.concatenate([f[:, None] for f in filters], axis=1)
    full_filter = [np.all(f) for f in filters]

    filtered_catalogue = catalogue.sel({list(catalogue.sizes.keys())[0]: full_filter})
    filtered_catalogue.attrs.update(filter_attrs)

    return filtered_catalogue


def filter_attr_str2list(string_attr):
    """
    In order to save the attributes to file, lists are converted to string.
    This function revert as string back to list

    Parameters
    ----------
    string_attr : str
        The original string

    Returns
    -------
    list_attr : list
        The parsed list
    """

    # Convert string to list
    if string_attr[0] == "[" and string_attr[-1] == "]":
        list_attr = string_attr[1:-1].split(", ")

    else:
        list_attr = string_attr.split(", ")

    return list_attr


def covariate_at_event(cov, catalogue, rate=False, rate_clip=None):
    """
    Helper function to get the value of a covariate at the event time and location

    Parameters
    ----------
    cov : xarray.DataArray
        The covariate to interpolate
    catalogue: xarray.Dataset
        Dataset containing earthquake magnitudes and earthquake locations
    rate : bool
        If true, return the cov_rate instead of the cov_value
    rate_clip : float, optional
        Rate at which to clip

    Returns
    -------
    value : xarray.DataArray
        Value or rate of the variable at the event.
    """
    # Historically, we use nearest-neighbour interpolation in space, and linear interpolation in time
    local = cov.interp({"x": catalogue.x, "y": catalogue.y}, method="nearest")

    if not rate:
        try:
            return local.interp({"time": catalogue.date_time})
        except ValueError:
            # If there is no time dimension in the covariate, interpolation is already done
            return local

    # If we want a rate, we use this manual way to determine a slope
    t_ind_after = (cov.time > catalogue.date_time).argmax(dim='time')
    t_ind_before = t_ind_after - 1
    delta_t = ((cov.time[t_ind_after] - cov.time[t_ind_before]) / 1e9).astype(float) / 31557600.0
    output = (local.isel(time=t_ind_after) - local.isel(time=t_ind_before)) / delta_t
    if rate_clip is not None:
        output = output.clip(min=rate_clip)

    return output
