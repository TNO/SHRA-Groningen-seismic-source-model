import tqdm
import xarray as xr


def rate_type_isotach_compaction_model(
        loading: xr.DataArray,
        cm_grid: xr.DataArray,
        params: dict,
        time_queries=None,
        initial_strains=None,
        output_rates=False
):
    # handle parameters
    load_rate_ref = -params['sigma_rate_ref']
    b_exp = params['b_exponent']
    cm_d = cm_grid * params['factor_cm_d']
    cm_ref = cm_grid * params['factor_cm_ref']

    # handle time dimension
    if time_queries:
        loading = loading.interp({'time': time_queries})
    dt = loading.time.values[1:] - loading.time.values[:-1]

    # handle initial conditions
    load_ref = loading.isel({'time': 0}, drop=True)
    zero_array = xr.zeros_like(load_ref)
    if isinstance(initial_strains, xr.Dataset):
        eps_e = initial_strains['elastic_strain']
        eps_i = initial_strains['plastic_strain']
    elif initial_strains is None:
        eps_e = zero_array
        eps_i = zero_array
    else:
        raise TypeError('provide initial strains as an xr.Dataset')

    # handle output and add initial conditions
    list_e = [eps_e]
    list_i = [eps_i]
    labels = ['elastic_strain', 'inelastic_strain']
    # strain rates are differentiated towards lower label, to maintain consistency with elastic solution
    list_edot = []
    list_idot = []
    if output_rates:
        labels = labels + [l + '_rate' for l in labels]

    # let's go!
    desc = 'calculating strain'
    for i in tqdm.tqdm(range(len(dt)), desc=desc):
        load = loading.isel({'time': i + 1}, drop=True)
        prev_load = loading.isel({'time': i}, drop=True)

        cm = (eps_e + eps_i + cm_ref * load_ref) / load
        dot_eps_i = load_rate_ref * (cm - cm_d) * (cm / cm_ref) ** (-1.0 / b_exp)
        eps_i = eps_i + dot_eps_i * dt[i]
        eps_e = cm_d * (load - load_ref)

        dot_eps_e = cm_d * (load - prev_load) / dt[i]

        list_e.append(eps_e)
        list_i.append(eps_i)
        if output_rates:
            list_edot.append(dot_eps_e)
            list_idot.append(dot_eps_i)

    list_out = [list_e, list_i]
    if output_rates:
        list_edot.append(xr.zeros_like(dot_eps_e))
        list_idot.append(xr.zeros_like(dot_eps_i))
        list_out = list_out + [list_edot, list_idot]
    list_out = [xr.concat(l, dim='time') for l in list_out]
    out = xr.Dataset(dict(zip(labels, list_out))).assign_coords({'time':loading.time})

    return out
