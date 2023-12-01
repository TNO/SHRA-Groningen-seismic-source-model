"""
Functionality related to Bayesian inference.
"""

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

DASK_VERBOSE = True


def dask_compute(computation):
    """
    Perform the dask computation that was postponed so far

    Parameters
    ----------
    computation: xarray.DataArray 
        Array containing the delayed computation

    Returns
    -------
    res: xarray.DataArray
        The computation result
    
    """
    # DASK_VERBOSE variable is set globally
    if DASK_VERBOSE:
        with ProgressBar():
            res = computation.compute()
    else:
        res = computation.compute()
    return res


def magnitude_log_likelihood(fmd_terms, catalogue, dm, use_dask=False):
    """
    Calculate the log_likelihood of a model, given the b (and zeta) values and the observed catalgoue.

    Parameters
    ----------
    fmd_terms: xarray.Dataset
        Dataset containing 'b' and 'zeta', at the location of the events in the catalogue
    catalogue: xarray.Dataset
        Dataset containing earthquake magnitudes and earthquake locations
    dm: float
        Magnitude step size
    use_dask: bool, optional
        Toggle for parallel computation with dask

    Returns
    -------
    log_likelihood: xarray.DataArray
        Log-likelihood for each parameter set
    """

    def ll_func(bval, zetaval, mrange, mstep):
        return (
                np.log(bval + 1.5 * zetaval * 10 ** (1.5 * mrange))
                + np.log(np.log(10))
                + np.log(10 ** (-bval * mrange))
                - zetaval * (10 ** (1.5 * mrange) - 1)
                + np.log(mstep)
        ).sum(dim=mrange.dims[0])

    b, zeta = fmd_terms["b"], fmd_terms["zeta"]
    m = catalogue.magnitude - catalogue.minmag + 0.5 * dm

    if use_dask:
        b = b.chunk("auto")
        zeta = zeta.chunk("auto")
        m = m.chunk("auto")
        ll = ll_func(b, zeta, m, dm)
        ll = dask_compute(ll)
        return ll

    if b.nbytes / 1e9 < 3:  # Do in one step if it doesn't take a huge amount of memory (b less than 3 GB)
        ll = ll_func(b, zeta, m, dm)
    else:
        # Loop to save memory. We assume that looping over one dimension is enough
        # Find the biggest dimension that's not the magnitude dimension, and loop over that one
        dim_sizes = dict(zip(b.dims, b.shape))
        allowed_split_dims = [d for d in b.dims if d != m.dims[0]]
        associated_sizes = [dim_sizes[k] for k in allowed_split_dims]
        largest_split_dim = allowed_split_dims[associated_sizes.index(max(associated_sizes))]
        chunks = []
        for param_i in range(len(b[largest_split_dim])):
            b_chunk = b.isel({largest_split_dim: param_i})
            try:
                zeta_chunk = zeta.isel({largest_split_dim: param_i})
            except ValueError:
                # Zeta does not have that dimension
                zeta_chunk = zeta
            chunks.append(ll_func(b_chunk, zeta_chunk, m, dm))
        ll = xr.concat(chunks, dim=largest_split_dim)

    return ll


def poisson_log_likelihood(background, etas, use_loop=False, use_dask=False):
    """
    Calculate the Poissonian log-likelihood

    Parameters
    ----------
    background: xarray.Dataset
        Contains dataarrays integral and local_term for ll calculation
    etas: xarray.Dataset
        Contains dataarrays integral and local_term for ll calculation
    use_loop: bool, optional
        Solve ll function by looping (decrease memory use by a lot)
    use_dask: bool, optional
        Toggle for parallel computation with dask

    Returns
    -------
    log_likelihood: xarray.DataArray
        Log-likelihood for each parameter set
    """

    # We dynamically detect the sum dimension since the name changes depending on the catalogue source
    sum_dim = [dim for dim in background.local_term.dims if dim not in background.integral.dims]
    assert len(sum_dim) == 1, "Summing dimension must be exactly one of the integral dimensions: (event dimension)"

    if use_dask:
        background = background.chunk("auto")
        etas = etas.chunk("auto")
        log_likelihood = (np.log(background.local_term + etas.local_term)).sum(dim=sum_dim) - (
                background.integral + etas.integral
        )
        log_likelihood = dask_compute(log_likelihood)
        return log_likelihood

    if not use_loop:
        try:
            log_likelihood = (np.log(background.local_term + etas.local_term)).sum(dim=sum_dim) - (
                    background.integral + etas.integral
            )
        except MemoryError:
            print("Failed to solve ll function by broadcasting, using loop instead to save memory")
            use_loop = True

    if use_loop:
        log_likelihood = solve_chunked(background, etas)

    return log_likelihood


def solve_chunked(background, etas):
    """
    Solve Poissonian log-likelihood in chunks that fit in memory

    Parameters
    ----------
    background: xarray.Dataset
        Contains dataarrays integral and local_term for ll calculation
    etas: xarray.Dataset
        Contains dataarrays integral and local_term for ll calculation

    """
    # We dynamically detect the sum dimension since the name changes depending on the catalogue source
    sum_dim = [dim for dim in background.local_term.dims if dim not in background.integral.dims]
    assert len(sum_dim) == 1, "Summing dimension must be exactly one of the integral dimensions: (event dimension)"

    expected_dims = np.unique(list(background.local_term.dims) + list(etas.local_term.dims))
    dim_sizes = {
        **dict(zip(background.local_term.dims, background.local_term.shape)),
        **dict(zip(etas.local_term.dims, etas.local_term.shape)),
    }
    expected_array_shape = [dim_sizes[k] for k in expected_dims]
    expected_array_size = np.prod(expected_array_shape) * 8 / 1e9  # In GiB

    if expected_array_size < 0.5:  # We assume 500 MiB will fit into memory.
        chunk = (np.log(background.local_term + etas.local_term)).sum(dim=sum_dim) - background.integral - etas.integral
    else:
        # Find the biggest dimension to split
        allowed_split_dims = [d for d in expected_dims if d not in sum_dim]
        associated_sizes = [dim_sizes[k] for k in allowed_split_dims]
        largest_split_dim = allowed_split_dims[associated_sizes.index(max(associated_sizes))]
        chunk = []
        for i in range(dim_sizes[largest_split_dim]):
            try:
                sub_background = background.isel({largest_split_dim: i})
            except ValueError:
                sub_background = background
            try:
                sub_etas = etas.isel({largest_split_dim: i})
            except ValueError:
                sub_etas = etas

            chunk.append(solve_chunked(sub_background, sub_etas))
        chunk = xr.concat(chunk, dim=largest_split_dim)

    return chunk


def get_posterior_probability(log_likelihood, prior=1):
    """
    Calculate the probability of a sample being drawn from the given model.

    Parameters
    ----------
    log_likelihood : xr.DataArray
        The log of the probability of the sample being drawn from the model.
    prior : float or xr.DataArray
        The prior probability of the sample being drawn from the model.

    Returns
    -------
    xr.DataArray
        The probability of the sample being drawn from the model.
    """

    likelihood = np.exp(log_likelihood - log_likelihood.max())
    likelihood /= likelihood.sum()

    return prior * likelihood


def get_combined_posterior(*args, fast=True, use_dask=False):
    """
    Combine several posteriors to get probability of shared parameters

    Parameters
    ----------
    *args : xr.DataArray
        All posterior probability distributions
    fast: boolean
        Collapse distributions before combining
    use_dask
        Use dask to manage computation

    Returns
    -------
    xr.DataArrays
        The posterior probability of parameters that are shared between all individual posterior distributions
    """

    all_dims = [a.dims for a in args]
    unique_dims = np.unique([item for items in all_dims for item in items])
    non_shared_dims = []
    for dim in unique_dims:
        dim_in_posterior = [dim in post.dims for post in args]
        if not np.all(dim_in_posterior):
            non_shared_dims.append(dim)

    if fast:
        sum_dims = [[d for d in non_shared_dims if d in arg.dims] for arg in args]
        summed_args = [arg.sum(dims) for arg, dims in zip(args, sum_dims)]
        combined = xr.dot(*summed_args, dims="...", optimize="greedy")
        combined /= combined.sum()
        return combined

    if use_dask:
        args = (a.chunk('auto') for a in args)
        combined = xr.dot(*args, dims=non_shared_dims, optimize="greedy")
        combined = dask_compute(combined)
    else:
        combined = xr.dot(*args, dims=non_shared_dims, optimize="greedy")

    combined /= combined.sum()
    return combined
