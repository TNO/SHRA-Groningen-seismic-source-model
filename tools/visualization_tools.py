import os
import calendar
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from tools.catalogue_tools import filter_attr_str2list
from matplotlib.ticker import MultipleLocator
# --------------------------------------------------
# functions for visualization of calibration results
# --------------------------------------------------
def plot_and_save_grid_evaluation(posterior: xr.DataArray, fig_path: Path, ignore_dims=None):

    """
    Plot the posterior probability distributions marginalised to all combinations of two dimensions, and marginalised
    to  one dimension.

    Parameters
    ----------
    posterior : xr.DataArray
        Array with posterior probability values
    log : (optional) list
        Optional list with dimensions of model parameters (as strings) that should be plotted on a log scale
    Returns
    -------

    """
    if ignore_dims is None:
        ignore_dims = ['dsm_model']

    max_post = posterior[np.unravel_index(posterior.argmax(), posterior.shape)]
    posterior = posterior.squeeze()
    ndim = len(posterior.dims)

    plt.figure(figsize=(3.5 * ndim, 3.5 * ndim), dpi=300)

    plt_i = 1
    for j, d_ref in enumerate(posterior.dims):
        for i, d in enumerate(posterior.dims):
            if i < j:
                ax = plt.subplot(ndim, ndim, plt_i)
                data = posterior.sum(dim=[x for x in posterior.dims if x != d and x != d_ref])
                data.T.plot(ax=ax, add_colorbar=False)
                ax.plot(max_post[d].values, max_post[d_ref].values, marker='o', mfc='w', mec='k')
                ax.set_xlabel(d, fontsize=20)
                ax.set_ylabel(d_ref, fontsize=20)
                ax.set_title('')
                if j!=ndim-1:
                    ax.set_xlabel('')
                if i!=0:
                    ax.set_ylabel('')
            elif i==j:
                ax = plt.subplot(ndim, ndim, plt_i)
                data = posterior.sum(dim=[x for x in posterior.dims if x != d])

                dx = posterior.coords[d].values[1:] - posterior.coords[d].values[:-1]
                # dx = posterior.coords[d].values[1] - posterior.coords[d].values[0]
                bins = np.concatenate([
                    [posterior.coords[d].values[0] - dx[0] * 0.5],
                    posterior.coords[d].values[0] - dx[0] * 0.5 + dx.cumsum(),
                    [posterior.coords[d].values[-1] + dx[-1] * 0.5]
                ])
                # bins = np.linspace(posterior.coords[d].values[0] - 0.5 * dx,
                #                     posterior.coords[d].values[-1] + 0.5 * dx,
                #                     posterior.coords[d].size + 1)
                ax.hist(posterior.coords[d].values, bins=bins, weights=data)
                ax.plot(max_post[d].values, 0, marker='o', mfc='w', mec='k')
                ax.set_xlabel(d, fontsize=20)
                ax.set_title('')
                if j!= ndim-1:
                    ax.set_xlabel('')
            plt_i += 1

    plt.savefig(fig_path, dpi=300)
    plt.close()
# -----------------------------------------------
# functions for visualization of forecast results
# -----------------------------------------------
def plot_and_save_annual_event_density_maps(forecast: xr.DataArray, fig_path: str, vis_epochs=None, gasyear=True):
    """
    Create maps of the reservoir with the event density, for each timestep.
    Parameters
    ----------
    forecast : xr.DataArray
        Forecasted event rates.
    vis_epochs
        Optional. Specify which epochs should be plotted. If not given, all available epochs are plotted.
    gasyear : bool
        Toggle to use gasyears in labels.

    Returns
    -------

    """
    # get spatial surface area of grid cells
    x = forecast.x.values
    y = forecast.y.values
    cell_area = (x[1] - x[0]) * (y[1] - y[0])
    convert_to_km2 = 1e6 / cell_area

    # get epochs that should be plotted
    if vis_epochs is None:
        epochs = forecast.time.values
    else:
        available_epochs = forecast.time.values
        epochs = [epoch for epoch in vis_epochs if epoch in available_epochs]

    xlim, ylim = groningen_reservoir_map_limits()

    # get event rates for one magnitude and mmax bin, set them in the correct units
    event_density = forecast.sel(time=epochs).isel(magnitude=0, branch_mmax=-1) * convert_to_km2
    vmax = event_density.max().values

    # plot all panels in one go
    panels_per_column = 3
    if len(epochs) == 3 or len(epochs) == 4:
        panels_per_column = 2
    axes_mesh = event_density.plot(x='x', y='y', col='time', add_colorbar=True,
                               col_wrap=panels_per_column, sharex=False, aspect=0.82,
                               cmap='hot_r', vmin=0, vmax=vmax, cbar_kwargs={'aspect': 40, 'shrink': 0.6})
    # update layout color bar
    axes_mesh.cbar.set_label('annual event density [km\u207B\u00b2]')

    # update layout of each panel
    for i, year in enumerate(epochs):
        year = str(int(dateTimetoDecYear(year)))
        ax = axes_mesh.axes.flat[i]
        ax.axis('equal')
        ax.axis('off')
        plot_reservoir_outline(ax=ax, lw=1)
        plot_coast(ax=ax, lw=1)
        plot_cities(ax=ax, s=10, color='gray', add_name=False)
        if gasyear:
            ax.set_title("GY%s/%s" % (year, str(int(year) + 1)), fontdict={'fontsize': 12})
        else:
            ax.set_title("%s" % year, fontdict={'fontsize': 12})
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # add overview map if there's space
    if len(epochs) < len(axes_mesh.axes.flat):
        ax = axes_mesh.axes.flat[len(epochs)]
        ax.axis('equal')
        ax.axis('off')
        plot_reservoir_outline(ax=ax, lw=1)
        plot_coast(ax=ax, lw=1)
        plot_cities(ax=ax, s=12, fontsize=10, color='k')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_frame_on(False)
        ax.set_visible(True)
    
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_and_save_annual_magnitude_model(forecast: xr.DataArray, basedir: str, gasyear=True):
    """
    Create and save figures of the forecasted magnitude-frequency relation, for each timestep.
    Parameters
    ----------
    forecast : xr.DataArray 
        Forecasted event rates.
    basedir : str
        path to directory where figures are written to.
    gasyear : bool
        Toggle to use gasyears in labels.

    Returns
    -------

    """

    ccfmd = forecast.sum(dim=['x', 'y'])
    time = np.array([a for a in forecast.time.values])
    mrange = forecast.magnitude.values
    mmax_array = forecast.branch_mmax.values

    for index_year, year in enumerate(time):
        year = dateTimetoDecYear(year)

        for im, mmax in enumerate(mmax_array):
            plt.semilogy(
                mrange,
                ccfmd.isel(time=index_year, branch_mmax=im),
                label='Mmax: {:.1f}'.format(float(mmax))
            )
        plt.legend()
        plt.xlim([mrange[0], mrange[-1] + 1])
        plt.ylim([1e-5, 30])
        plt.xlabel('Magnitude')
        plt.ylabel('Nr events exceeding M')

        if gasyear:
            plt.title("GY%s/%s" % (year, str(int(year) + 1)), fontdict={'fontsize': 12})
            plt.title('Fieldwide forecast for {0:}/{1:}\n Total number of events: {2:.2f}'.format(
                int(year),
                str(int(year) + 1),
                float(ccfmd.isel(time=index_year, branch_mmax=im, magnitude=0)))
            )
        else:
            plt.title('Fieldwide forecast for {0:}\n Total number of events: {1:.2f}'.format(
                year,
                float(ccfmd.isel(time=index_year, branch_mmax=im, magnitude=0)))
            )

        if os.path.isdir(basedir) is False:
            os.mkdir(basedir)

        plt.savefig(os.path.join(basedir, "fieldwide_frequency_magnitude_distribution_{}.png".format(int(year))),
                    dpi=300)
        plt.close()

def bin_observed_events(eq_catalogue, modelled_time_intervals, incomplete_intervals=False):
    """
    Bin the observed earthquake catalogue in the requested time intervals.

    Parameters
    ----------
    path_to_eq_catalogue : str or list
        String or list of strings to earthquake catalogue .h5 file(s). Multiple files may be given, for instance the
        training and testing catalogues used for ssm calibration. Catalogue .h5 files should be generated with the
        parse_input.py function in this repo.
    incomplete_intervals : bool
        If False, data from incomplete time intervals are not returned. Incomplete time intervals are determined from
        the 'date_filter' attribute in the catalogue .h5 files.
    modelled_time_intervals : np.array
        Array containing the modelled time intervals.

    Returns
    -------
    observed_count : np.array
        Array containing the number of observed events per time interval
    bin_edges : np.array
        Array containing the edges of the time interval. NOTE: length is len(observed_count) + 1
    catalogue_start : float
    catalogue_end : float

    """

    catalogue_start, catalogue_end = filter_attr_str2list(eq_catalogue.date_filter)
    catalogue_start = pd.to_datetime(catalogue_start, format=r"%Y%m%d")
    catalogue_end = pd.to_datetime(catalogue_end, format=r"%Y%m%d") + pd.DateOffset(days=1) - pd.DateOffset(seconds=1)

    # get observed counts per modelled time interval and set time intervals without events to np.nan
    observed_count, bin_edges = np.histogram(eq_catalogue.timestamp.date_time.values, modelled_time_intervals)
    observed_count = np.where(observed_count < 1, np.nan, observed_count)

    # if only complete years are desired:
    if incomplete_intervals is False:
        lower_bin_edges = bin_edges[:-1]
        upper_bin_edges = bin_edges[1:]
        time_mask = np.logical_and(
            lower_bin_edges > catalogue_start,
            upper_bin_edges < catalogue_end
        )
        observed_count[~time_mask] = np.nan

    return observed_count, bin_edges, catalogue_start, catalogue_end


def plot_and_save_fieldwide_event_counts(rate_forecast, event_count_uncertainty, calibration_catalogue_path, fig_path,
                                         alternative_catalogue_path=None, plot_incomplete_intervals=False, reference_rate=None):
    """
    Produce a figure of the fieldwide observed and modelled event counts versus time.

    Parameters
    ----------
    rate_forecast : xr.DataArray
        Model forecast results for event rates.
    event_count_uncertainty : xr.DataArray
        Model forecast results for the event count uncertainty.
    calibration_catalogue_path : xr.DataArray
        String of earthquake catalogue .h5 file used for the model calibration.
    alternative_catalogue_path
        Optional. xr.DataArray
    plot_incomplete_intervals : bool
        Flag, if True the data in time intervals not entirely covered by the catalogues will be plotted as well.
    Returns
    -------

    """

    # observed event counts from calibration catalogue(s), used for reproducing SDRA figures
    counts_calib_catalogue, intervals_calib_catalogue, start_calib, end_calib = bin_observed_events(
        calibration_catalogue_path,
        rate_forecast.time.values
    )

    if alternative_catalogue_path:
        # overwrite with alternative catalogue
        counts_calib_catalogue, intervals_calib_catalogue, _, _ = bin_observed_events(
            alternative_catalogue_path,
            rate_forecast.time.values
        )


    # Get modelled mean event counts
    round_mean = rate_forecast.sum(dim=['x', 'y']).round()


    # Get modelled event uncertainty
    cumulative_event_count_uncertainty = event_count_uncertainty.cumsum(dim='nr_events').transpose('time', 'nr_events')
    lower = np.argmax(cumulative_event_count_uncertainty.values[:-1] > 0.025, axis=1)
    upper = np.argmax(cumulative_event_count_uncertainty.values[:-1] > 0.975, axis=1)
    lower = np.asarray([[nr, nr] for nr in lower]).flatten()
    upper = np.asarray([[nr, nr] for nr in upper]).flatten()

    # determine ymax
    ymax = max(50, upper.max() + 5)

    ### start plotting ###
    _, ax = plt.subplots(1, 1)
    round_mean.plot.step(where='post', c='grey', label='simulated')

    if reference_rate is not None:
       round_ref = reference_rate.sum(dim=['x', 'y']).round() 
       round_ref.plot.step(where='post', c='grey', label='reference', lw=0.75, ls=':')


    plt.step(intervals_calib_catalogue[:-1], counts_calib_catalogue, where='post', lw=2, c='k', label='observed', zorder=10)

    # plot uncertainty with custom step function
    model_plot_years = np.asarray([[t,t] for t in rate_forecast.time.values]).flatten()[1:-1]
    # final_year = model_plot_years[-1] + (model_plot_years[1] - model_plot_years[0])
    # model_plot_years = np.concatenate((model_plot_years, [final_year]))
    final_year = model_plot_years[-1]
    ax.fill_between(model_plot_years, lower, upper, facecolor='silver', label='95% confidence bounds')
    ax.fill_between([start_calib, end_calib], [0, 0], [ymax, ymax], facecolor=(0.0, 0.5, 0.1, 0.3), label='model calibration period')

    # layout
    ax.legend(loc=1)
    xtick_major = np.arange(1995, 2100, 5)
    xtick_minor = np.arange(1995, 2100, 1)
    xtick_major = [pd.to_datetime(a, format=r'%Y') for a in xtick_major]
    xtick_minor = [pd.to_datetime(a, format=r'%Y') for a in xtick_minor]
    ytick_major = np.arange(0, ymax + 5, 5.0)
    for x in xtick_major:
        ax.plot((x, x), (0, ymax), '--', color="dimgray", lw=0.25)
    for y in ytick_major:
        ax.plot((0, float(final_year)), (y, y), '--', color="dimgray", lw=0.25)
    ax.set_ylabel('Number of events per year')
    ax.set_xlabel('Year')
    ax.set_xticks(xtick_major)
    ax.set_xticks(xtick_minor, labels=['' for _ in xtick_minor], minor=True)
    ax.set_yticks(ytick_major)
    ax.xaxis.set_major_formatter(DateFormatter(r'%Y'))
    ax.yaxis.set_tick_params(which='minor', bottom=False)
    ax.set_xlim([pd.to_datetime(1995, format=r'%Y'), final_year])
    ax.set_ylim([0, ymax])
    plt.title('')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

def plot_fieldwide_event_counts(rate_forecast, event_count_uncertainty, calibration_catalogue_path,
                                         alternative_catalogue_path=None, plot_incomplete_intervals=False):
    """
    Produce a figure of the fieldwide observed and modelled event counts versus time.

    Parameters
    ----------
    rate_forecast : xr.DataArray
        Model forecast results for event rates.
    event_count_uncertainty : xr.DataArray
        Model forecast results for the event count uncertainty.
    calibration_catalogue_path : str
        String of earthquake catalogue .h5 file used for the model calibration.
    alternative_catalogue_path
        Optional. String or list of alternative earthquake catalogues in .h5 file formats.
    plot_incomplete_intervals : bool
        Flag, if True the data in time intervals not entirely covered by the catalogues will be plotted as well.
    Returns
    -------

    """

    # observed event counts from calibration catalogue(s), used for reproducing SDRA figures
    counts_calib_catalogue, intervals_calib_catalogue, start_calib, end_calib = get_binned_event_counts_observed(
        calibration_catalogue_path,
        rate_forecast.time.values
    )
    intervals_calib_catalogue = np.asarray([[float(t), float(t)] for t in intervals_calib_catalogue]).flatten()[1:-1]
    counts_calib_catalogue = np.asarray([[nr, nr] for nr in counts_calib_catalogue]).flatten().astype(float)

    # observed event counts from alternative catalogues (optional)
    if plot_incomplete_intervals is True and alternative_catalogue_path is None:
        # alternative catalogue is calibration catalogue with incomplete time intervals
        alternative_catalogue_path = calibration_catalogue_path
    if alternative_catalogue_path is not None:
        counts_alt_catalogue, intervals_alt_catalogue, _, _ = get_binned_event_counts_observed(
            alternative_catalogue_path,
            rate_forecast.time.values,
            incomplete_intervals=plot_incomplete_intervals
        )
        intervals_alt_catalogue = np.asarray([[float(t), float(t)] for t in intervals_alt_catalogue]).flatten()[1:-1]
        counts_alt_catalogue = np.asarray([[nr, nr] for nr in counts_alt_catalogue]).flatten().astype(float)

    # Get modelled mean event counts
    round_mean = rate_forecast.sum(dim=['x', 'y']).round()
    model_plot_annual_events = np.asarray([[nr, nr] for nr in round_mean]).flatten()
    model_plot_years = np.asarray([[dateTimetoDecYear(t), dateTimetoDecYear(t)] for t in rate_forecast.time.values]).\
                           flatten()[1:]
    final_year = np.array([ dateTimetoDecYear(rate_forecast.time.values[-1]) + (model_plot_years[1]-model_plot_years[0])])
    model_plot_years = np.concatenate((model_plot_years, final_year))   # add final year for plotting

    # Get modelled mean event count Poissonian uncertainty and inflate to roughly represent ETAS
    cumulative_event_count_uncertainty = event_count_uncertainty.cumsum(dim='nr_events').transpose('time', 'nr_events')
    lower = np.argmax(cumulative_event_count_uncertainty.values > 0.025, axis=1)
    upper = np.argmax(cumulative_event_count_uncertainty.values > 0.975, axis=1)
    lower = np.asarray([[nr, nr] for nr in lower]).flatten()
    upper = np.asarray([[nr, nr] for nr in upper]).flatten()

    # determine ymax
    ymax = 50
    if upper.max() > 50:
        ymax = upper.max() + 5

    ### start plotting ###
    fig, ax = plt.subplots(1, 1)
    # plot simulated mean and observed data
    if alternative_catalogue_path is not None:
        ax.plot(intervals_alt_catalogue, counts_alt_catalogue, c='k', lw=1.5, zorder=10)
    ax.plot(intervals_calib_catalogue, counts_calib_catalogue, lw=2, c='k', label='observed', zorder=10)
    ax.plot(model_plot_years, model_plot_annual_events, c='grey', label='simulated')

    # plot uncertainty
    ax.fill_between(model_plot_years, lower, upper, facecolor='silver', label='95% confidence bounds')
    ax.fill_between([start_calib, end_calib], [0, 0], [ymax, ymax], facecolor=(0.0, 0.5, 0.1, 0.3), label='model calibration period')

    # layout
    ax.legend()
    xtick_major = np.arange(1995, final_year + 5, 5.0)
    ytick_major = np.arange(0, ymax + 5, 5.0)
    for x in xtick_major:
        ax.plot((x, x), (0, ymax), '--', color="dimgray", lw=0.25)
    for y in ytick_major:
        ax.plot((0, float(final_year)), (y, y), '--', color="dimgray", lw=0.25)
    ax.set_ylabel('Number of events per year')
    ax.set_xlabel('Year')
    ax.set_xticks(xtick_major)
    ax.set_yticks(ytick_major)
    ax.minorticks_on()
    ax.yaxis.set_tick_params(which='minor', bottom=False)
    ax.set_xlim([1995, final_year])
    ax.set_ylim([0, ymax])


def get_binned_event_counts_observed(path_to_eq_catalogue, modelled_time_intervals, incomplete_intervals=False):
    """
    Bin the observed earthquake catalogue in the requested time intervals.

    Parameters
    ----------
    path_to_eq_catalogue : str or list
        String or list of strings to earthquake catalogue .h5 file(s). Multiple files may be given, for instance the
        training and testing catalogues used for ssm calibration. Catalogue .h5 files should be generated with the
        parse_input.py function in this repo.
    incomplete_intervals : bool
        If False, data from incomplete time intervals are not returned. Incomplete time intervals are determined from
        the 'date_filter' attribute in the catalogue .h5 files.
    modelled_time_intervals : np.array
        Array containing the modelled time intervals.

    Returns
    -------
    observed_count : np.array
        Array containing the number of observed events per time interval
    bin_edges : np.array
        Array containing the edges of the time interval. NOTE: length is len(observed_count) + 1
    catalogue_start : float
    catalogue_end : float

    """

    # handle single or multiple eq catalogues and extract calibration period before merging
    path_to_eq_catalogue = np.array(path_to_eq_catalogue, ndmin=1)
    earthquake_catalogue_list = [xr.load_dataset(cat, engine='h5netcdf') for cat in path_to_eq_catalogue]
    t = str(np.min([eval(cat.attrs['date_filter']) for cat in earthquake_catalogue_list]))
    catalogue_start = np.round(
        dateTimetoDecYear(np.datetime64(f'{t[:4]}-{t[4:6]}-{t[6:]}T00:00:00.000000000')),
        decimals=5
    )
    t = str(np.max([eval(cat.attrs['date_filter']) for cat in earthquake_catalogue_list]))
    catalogue_end = np.round(
        dateTimetoDecYear(np.datetime64(f'{t[:4]}-{t[4:6]}-{t[6:]}T23:59:59.999999999')),
        decimals=5
    )
    earthquake_catalogue = xr.merge(earthquake_catalogue_list)

    # get observed counts per modelled time interval and set time intervals without events to np.nan
    observed_events_timing = [dateTimetoDecYear(t) for t in earthquake_catalogue.timestamp.date_time.values]
    modelled_timing = [dateTimetoDecYear(t) for t in modelled_time_intervals]
    observed_count, bin_edges = np.histogram(observed_events_timing, modelled_timing)
    observed_count = np.where(observed_count < 1, np.nan, observed_count)
    lower_bin_edges = bin_edges[:-1]
    upper_bin_edges = bin_edges[1:]

    # if only complete years are desired:
    if incomplete_intervals is False:
        time_mask = np.logical_and(
            lower_bin_edges > catalogue_start,
            upper_bin_edges < catalogue_end
        )
        observed_count[~time_mask] = np.nan

    return observed_count, bin_edges, catalogue_start, catalogue_end


# -------------------------------
# general visualization functions
# -------------------------------

def plot_coast(ax: plt.axes=plt.gca(), color: str= 'grey', lw: float=1):
    """
    Plot outline of Dutch coastline.

    Parameters
    ----------
    ax : bool or plt.axes
        Optional. Handle to axis object, if not given plotted to currently active axis
    color : str
        Optional. Color of outline. Default is grey
    lw : float
        Optional. Line width.
    Returns
    -------

    """

    file = 'coast_outline.csv'
    res_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../res')
    outline = np.genfromtxt(os.path.join(res_dir, file), names=True, delimiter=',')

    if not ax:
        plt.plot(outline['x'], outline['y'], c=color, lw=lw, zorder=4)
    else:
        ax.plot(outline['x'], outline['y'], c=color, lw=lw, zorder=4)

def plot_cities(ax: plt.axes=plt.gca(), symbol: str= 's', s: int=15, color: str= 'grey', fontsize: float=8,
                add_name: bool=True):
    """
    Plot locations of populations centre on and near the Groningen reservoir

    Parameters
    ----------
    ax : plt.axes
        Optional. Provide axis to plot on, otherwise current axis is used or new axis is created
    symbol : str
        Optional. Symbol shape, square by default. See matplotlib documentation
    s : int
        Optional. Marker size, 15 by default.
    color : str
        Optional. Color of markers.
    fontsize : float
        Optional. Font size for place names, if add_name is True.
    add_name : bool
        If true, plot names of population centres as well.

    Returns
    -------

    """

    source_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../res', 'centres_of_population.csv')
    cities = np.genfromtxt(source_path, names=True, delimiter=',', dtype='f8,f8,U50')
    for city in cities:
        x_offset = 3000
        y_offset = 3000
        if city['plaats'] == 'Delfzijl':
            x_offset = 6500
            y_offset = -2000
        if city['plaats'] == 'Hoogezand':
            y_offset = -2000
        if city['plaats'] == 'Ten Boer':
            x_offset = -500
        if city['plaats'] == 'Groningen':
            x_offset = 8000
        if city['plaats'] == 'Winschoten':
            x_offset = 8000
        if city['plaats'] == 'Loppersum':
            x_offset = 7000
        if not ax:
            plt.scatter(city['x'], city['y'], marker=symbol, s=s, color=color, zorder=5)
            if add_name:
                plt.text(city['x'] - x_offset, city['y'] - y_offset, city['plaats'],
                         color=color, fontsize=fontsize, zorder=5)
        else:
            ax.scatter(city['x'], city['y'], marker=symbol, s=s, color=color)
            if add_name:
                ax.text(city['x'] - x_offset, city['y'] - y_offset, city['plaats'],
                        color=color, fontsize=fontsize, zorder=5)

def plot_reservoir_outline(ax: plt.axes=plt.gca(), color: str='k', lw: float=2):
    """
    Plot reservoir outline.

    Parameters
    ----------
    ax : plt.axes
        Optional. Axis object to plot outline on. If not given, current axis is used or a new one created.
    color : str
        Optional. Color for outline.
    lw : float
        Optional. Line width for contour.

    Returns
    -------

    """

    source_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../res', 'Groningen_field_outline.csv')
    outline = np.genfromtxt(source_path, names=True, delimiter=',')
    if ax == 'plt':
        plt.plot(outline['x'], outline['y'], c=color, lw=lw, zorder=6)
    else:
        ax.plot(outline['x'], outline['y'], c=color, lw=lw, zorder=6)

def groningen_reservoir_map_limits():
    """
    Provide limits of the groningen reservoir for plotting purposes. Limits in rijksdriehoekcoordinaten system.

    Returns
    -------
    map_xlim : list

    map_ylim : list

    """

    map_xlim = [225000, 275000]
    map_ylim = [560000, 614000]

    return map_xlim, map_ylim

def dateTimetoDecYear(datetime):
    """
    Take a year in numpy.datetime64 format representation and output decimal notation (e.g. 2016.82764). Used for
    plotting time series.
    :param datetime:
    :return: Year in numpy datetime64 format
    """

    dtstring = np.datetime_as_string(datetime)
    year = float(dtstring[0:4])
    month = float(dtstring[5:7])
    day = float(dtstring[8:10])
    hour = float(dtstring[11:13])
    minute = float(dtstring[14:16])
    second = float(dtstring[17:19])
    nonleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthdays = 0
    if calendar.isleap(year):
        daysPerYear = 366.0
        for a in range(0, int(month) - 1):
            monthdays = monthdays + leap[a]
    else:
        daysPerYear = 365.0
        for a in range(0, int(month) - 1):
            monthdays = monthdays + nonleap[a]

    hoursPerDay = 24.
    minutesPerDay = 1440.
    secondsPerDay = 86400.

    day = monthdays + day + (hour / hoursPerDay) + (minute / minutesPerDay) + (second / secondsPerDay) - 1
    # -1 so that 2016-01-01T00:00:00 gives 2016.0000
    return year + (day / daysPerYear)

def write_field_csv(forecast, logic_tree_weights, csv_path):

    try:
        mmax_weights = logic_tree_weights['branch_mmax']
    except KeyError:
        mmax_weights = [0.27, 0.405, 0.1875, 0.1075, 0.025, 0.005]
        mmax_weights = xr.DataArray(mmax_weights, coords={'branch_mmax':[4.0, 4.5, 5.0, 5.5, 6.0, 6.5]})

    forecast = xr.dot(forecast, mmax_weights).sum(dim=['x', 'y'])
    times = pd.to_datetime(forecast.time)
    if times[0].month == 10:
        # Input is in gasyears
        times = np.array([f'GY{t.year}/{t.year+1}' for t in times])[:,None]
    else:
        times = np.array([f'{t.year}' for t in times])[:,None]
    
    rate = np.round(forecast.interp(magnitude=forecast.magnitude[0]).values[:,None],2)
    mags = [3.5, 3.6, 4.0, 4.5, 5.0]
    prob = 100*(1.0 - np.exp(-forecast.interp(magnitude=mags).values))
    prob = np.asarray([[f'{a:.2f}%' for a in b] for b in prob])

    output = pd.DataFrame(np.concatenate([times, rate, prob], axis=-1), columns = ['Time', 'rate']+[f'M{a}' for a in mags])
    output.to_csv(csv_path, index=False)
