"""
This module contains functions for fault sqlite file reading/parsing
"""

import math
import sqlite3
import numpy as np
import xarray as xr
import rioxarray  # pylint: disable=unused-import

def _query_sql_dbase(path, sql_query):
    """
    Send query to sqlite database and return response

    Parameters
    ----------
    path : str
        Path to the raw data file
    sql_query: str
        Query to send to database

    Returns
    -------
    data : xarray.Dataset
        Parsed query response
    """
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(sql_query)

    firstrow = cursor.fetchone()
    types = ["float" for _ in firstrow]
    varname = [i[0] for i in cursor.description]

    dtype = list(zip(varname, types))
    data = np.fromiter(cursor, dtype=dtype)
    data = np.insert(data, 0, firstrow)

    conn.close()

    data = xr.Dataset({n.lower(): xr.DataArray(data[n], name=n.lower()) for n in data.dtype.names})

    return data


def _add_grad_l_r_t(data):
    """
    Add gradient (grad), representative length (l), throw thickness ratio (r) and thickness (t) to the dataset
    Remove properties that won't be needed afterwards: fault_id, pillar_id, thickness_f, thickness1_f,
                                                       thickness_h, thickness1_h

    Parameters
    ----------
    data : xarray.Dataset
        Dataset with properties from the raw data file
    Returns
    -------
    updated_dataset: xarray.Dataset
        Dataset with updated properties

    """
    # We use the rather strange 'average' thickness define by Bourne and Oates (half of what I'd call avg thickness)
    data["thickness"] = 0.25 * (data["thickness_f"] + data["thickness1_f"] + data["thickness1_h"] + data["thickness_h"])
    data["grad"] = data["offset"]
    data["r"] = data["grad"] / data["thickness"]

    # Create array of representative lengths
    # We loop over faults to properly accomodate fault transitions
    length = []
    faults = np.unique(data["fault_id"])
    for fault in faults:
        faultdata = data.sel({list(data.dims.keys())[0]: data["fault_id"] == fault})
        xfault = faultdata["x"]
        yfault = faultdata["y"]

        midx = 0.5 * (xfault[0:-1] + xfault[1:])
        midy = 0.5 * (yfault[0:-1] + yfault[1:])
        startdist = float(math.sqrt((xfault[0] - midx[0]) ** 2 + (yfault[0] - midy[0]) ** 2))
        enddist = float(math.sqrt((xfault[-1] - midx[-1]) ** 2 + (yfault[-1] - midy[-1]) ** 2))
        middist = (
            np.sqrt((midx[0:-1] - xfault[1:-1]) ** 2 + (midy[0:-1] - yfault[1:-1]) ** 2)
            + np.sqrt((midx[1:] - xfault[1:-1]) ** 2 + (midy[1:] - yfault[1:-1]) ** 2)
        ).values.tolist()
        length.append(startdist)
        length.extend(middist)
        length.append(enddist)
    length = xr.DataArray(length, name="length")
    data["length"] = length

    # Remove properties that won't be needed afterwards:
    updated_data = data.drop_vars(
        ["fault_id", "pillar_id", "thickness_f", "thickness1_f", "thickness_h", "thickness1_h", "offset"]
    )

    return updated_data


def get_faults(path):
    """
    Read the fault data at path and add 'standard' attributes that are needed later

    Parameters
    ----------
    path : str
        Path to the raw data file

    Returns
    -------
    data : xarray.Dataset
        Parsed data in dataset format
    """
    # Support different database structures
    try:
        sql = (
            "SELECT FAULT_ID,PILLAR_ID,X,Y,Offset,Thickness_f,Thickness_h,Thickness1_f,Thickness1_h,Dip,Azimuth"
            " FROM pillar_geom ORDER BY FAULT_ID, PILLAR_ID"
        )
        parsed_data = _query_sql_dbase(path, sql)

    except ValueError:
        sql = (
            "SELECT FAULT_ID,PILLAR_ID,X,Y,Offset,Thickness_f,Thickness_h,Dip,Azimuth FROM pillar_geom ORDER BY"
            " FAULT_ID, PILLAR_ID"
        )
        parsed_data = _query_sql_dbase(path, sql)

    data = _add_grad_l_r_t(parsed_data)
    data.rio.write_crs(28992, inplace=True)

    return data
