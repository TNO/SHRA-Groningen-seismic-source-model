"""
This module contains functions for polygon reading/parsing and general operations using polygons
"""

import pandas as pd
import geopandas as gpd
import shapely


def get_polygon_gdf_from_file(polygon_path, field_id="GroningenFieldGWC"):
    """
    Reads a csv and returns a GeoDataFrame with the polygon

    Parameters
    ----------
    polygon_path : str
        Path to the csv

    Returns
    -------
    polygon_gdf : geopandas Dataframe
        The dataframe containing the polygon
    """
    outline = pd.read_csv(polygon_path)
    polygon = shapely.geometry.Polygon(zip(outline.x, outline.y))
    polygon_data = {"field_id": [field_id], "polygon": [polygon]}
    polygon_gdf = gpd.GeoDataFrame(data=polygon_data, geometry="polygon", crs=28992).set_index("field_id")
    polygon_gdf.attrs["field_id"] = field_id

    return polygon_gdf


def get_polygon_gdf_from_points(x, y):
    """
    Reads a csv and returns a GeoDataFrame with the polygon

    Parameters
    ----------
    x: array-like
        Contains the x coordinates of the polygon
    y: array-like
        Contains the y coordinates of the polygon

    Returns
    -------
    polygon_gdf : geopandas Dataframe
        The dataframe containing the polygon
    """
    polygon = shapely.geometry.Polygon(zip(x, y))
    polygon_data = {"polygon": [polygon]}
    polygon_gdf = gpd.GeoDataFrame(data=polygon_data, geometry="polygon", crs=28992)

    return polygon_gdf


def filter_catalogue_w_polygon(catalogue, polygon, inside=True):
    """
    Filters the catalogue by a polygon

    Parameters
    ----------
    catalogue : xarray Dataset
        The catalogue to filter
    polygon : geopandas DataFrame
        The polygon to filter by
    inside : bool
        If True, only the records that are inside the polygon are returned.
        If False, only the records that are outside the polygon are returned.

    Returns
    -------
    catalogue : xarray Dataset
        The filtered catalogue
    """
    points = gpd.GeoSeries(gpd.points_from_xy(catalogue["x"], catalogue["y"], crs=28992))
    if inside:
        # Keep points inside the polygon
        keep = [polygon.contains(p).values[0] for p in points]
    else:
        # Keep points outside the polygon
        keep = [not polygon.contains(p).values[0] for p in points]

    return catalogue.sel({list(catalogue.dims.keys())[0]: keep})
