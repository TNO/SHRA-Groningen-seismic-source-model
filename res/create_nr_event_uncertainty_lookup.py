"""This file can be used to recreate the event_uncertainty_with_etas.h5 file that is included in the repository"""

import time
import numpy as np
import xarray as xr
from scipy.stats import poisson
from tqdm import tqdm
from chaintools.chaintools import tools_xarray as xf

def build_pmf_including_aftershocks(backgroundrate, r):
    """
    Build the indiviual probability mass functions for observing a number of events in each generation of aftershocks

    Parameters
    ----------
    backgroundrate: float
        The number of events expected as background rate
    r: float
        The expected (mean) number of aftershocks in the next generation for a single mainshock

    Returns
    -------
    gen_list: list of np.ndarrays
        A list where each entry is the probability mass function of the number of events in that generation
    """

    # The expected mean number of events for the mainshocks and all generations combined (analytical expectation value)
    expected_mean = backgroundrate * (1.0 / (1.0 - r))
    x = np.arange(max(int(2 * expected_mean), 50))

    # Create the pmf. Entry i,j gives the probability of observing j events in the current generation, given that i
    # events were observed in the previous generation
    expectation_pmf = np.zeros((len(x), len(x)))
    for i, nr_shocks_prev in enumerate(x):
        expectation_pmf[i, :] = poisson.pmf(x, x[i] * r)

    # Build pmfs for each generation until probability of 1 or more aftershocks in final generation is <0.1%
    pmf_main = poisson.pmf(x, backgroundrate)
    gen_list = [pmf_main]
    while gen_list[-1][0] < 0.999:
        pmf_next = nr_shocks_current_gen(gen_list[-1], expectation_pmf)
        gen_list.append(pmf_next)

    return gen_list


def nr_shocks_current_gen(pmf_prev_gen, expectation_pmf):
    """
    Calculate the probability mass function for this generation, given the pmf of events in the previous generation

    Parameters
    ----------
    pmf_prev_gen: np.ndarray
        Probability mass function of the number of events in the previous generation
    expectation_pmf: np.ndarray
        The probability of observing j events in the current generation, given that i events were observed in
        the previous generation

    Returns
    -------
    pmf_new: np.ndarray
        Probability mass function for this generation
    """
    pmf_new = np.sum(pmf_prev_gen[:, None] * expectation_pmf, axis=0)
    return pmf_new


def combine_generations(gen_list):
    """
    Add pmfs together to get pmf of total number of events in the full mainshock+aftershock sequence

    Parameters
    ----------
    gen_list: list of np.ndarrays
        A list where each entry is the probability mass function of the number of events in that generation

    Returns
    -------
    comb_pmf: np.ndarray
        Pmf of total number of events in the full mainshock+aftershock sequence
    """

    comb_pmf = gen_list[0]
    offsets = np.linspace(len(comb_pmf) - 1, 0, len(comb_pmf)).astype(int)
    for i in range(1, len(gen_list)):
        pmf_gen = gen_list[i][None, :]
        comb_pmf = np.fliplr(comb_pmf[:, None] * pmf_gen)
        res = np.zeros(len(offsets))
        for j, offset in enumerate(offsets):
            res[j] = np.trace(comb_pmf, offset)
        comb_pmf = res

    return comb_pmf


def main_func():
    """
    Create a lookup table for the annual uncertainty in the number of events including ETAS effects
    """
    start_time = time.time()

    # Create the table
    rates = np.linspace(0, 100.0, 201)
    r_range = np.linspace(0.0, 0.5, 101)

    result_pmf_matrix = np.zeros((len(rates), len(r_range), 400))
    for i, rate in enumerate(tqdm(rates, desc="Creating nr_event lookup table for use with ETAS")):
        for j, r in enumerate(r_range):
            gen_list = build_pmf_including_aftershocks(rate, r)
            final_pmf = combine_generations(gen_list)
            result_pmf_matrix[i, j, : len(final_pmf)] = final_pmf

    pmf = xr.DataArray(
        result_pmf_matrix,
        coords={"background_rate": rates, "r": r_range, "nr_events": np.arange(result_pmf_matrix.shape[2])},
    )

    # Save result to file
    config = {"data_sinks":{"event_uncertainty":{"path": "event_uncertainty_with_etas.h5"}}}
    xf.store(pmf, "event_uncertainty", config)
    print(f"Total runtime: {(time.time() - start_time) / 60.:.2f} minutes")


if __name__ == "__main__":
    main_func()
