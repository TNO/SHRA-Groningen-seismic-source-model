## Introduction ##

This documentation is published together with the codebase for the 'TNO Modelchain for Groningen' (or 'modelchain' for short). <br>
The modelchain code allows you to perform a Seismic Hazard and Risk Analysis (SHRA) for the Groningen area in the Netherlands. <br>
The code is written in Python and is published through the following repositories:  
- [`SHRA-Groningen-seismic_source_model`](https://github.com/TNO/SHRA-Groningen-seismic-source-model)
- [`SHRA-Groningen-hazard_risk_models`](https://github.com/TNO/SHRA-Groningen-hazard-risk-models)
- [`SHRA-Groningen-chaintools`](https://github.com/TNO/SHRA-Groningen-chaintools)

All these repositories are licensed under the [EUPL](/LICENSE).

## Included models ##

This document is meant to assist code users and developers to get SHRA calculations running on their machines. <br>
It does not contain extensive background documentation of the models which are implemented, or on particular implementation choices. <br>
The body of literature on models for SHRA in Groningen is extensive. The models that are included in the modelchain are: <br>

- Seismological source model 'TNO-2020' (part of [`SHRA-Groningen-seismic_source_model`](https://github.com/TNO/SHRA-Groningen-seismic-source-model) repository) <br>
This model is largely based on the work by
[Bourne and Oates, 2017](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017JB014356), 
[Bourne and Oates, 2020](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JB020013), and updated based on 
recommendations by TNO as described in 
[TNO report R11961, 2022](https://www.nlog.nl/sites/default/files/2023-07/tno2022_r11961_sdra_status_rapport_2022_-_final_signed_gelakt.pdf).
<br>

- Rupture model (part of [`SHRA-Groningen-hazard_risk_models`](https://github.com/TNO/SHRA-Groningen-hazard-risk-models) repository)<br> 
This model takes care of the conversion of $R_{hyp}$ (distance to hypocentre of event) to $R_{rup}$ (distance to closest point on event's
rupture plane). This is needed for Ground motion models which are conditioned on $R_{rup}$. This model is based on an 
internal draft report from NAM (_Note for File: Finite rupture simulation for ground motion modelling and probabilistic 
seismic hazard and risk analysis for the Groningen gas field_)


- Ground motion models (part of [`SHRA-Groningen-hazard_risk_models`](https://github.com/TNO/SHRA-Groningen-hazard-risk-models) repository)
    - [V5, Bommer et al., 2018](https://nam-onderzoeksrapporten.data-app.nl/reports/download/groningen/en/52a1edec-6824-4ab3-8d92-3294c9cbec3a); 
    - [V6, Bommer et al., 2019](https://nam-onderzoeksrapporten.data-app.nl/reports/download/groningen/en/b66dd73e-9ff9-4be8-9302-5a2b514414bd); 
    - [V7, Bommer et al., 2021](https://nam-onderzoeksrapporten.data-app.nl/reports/download/groningen/en/96b7a3b0-98a7-4eaf-99bd-465b35f90ec8).

  For models V5 and V6, TNO recommendations regarding the period-to-period correlation structure at site-response level
according to [TNO report R11961, 2022](https://www.nlog.nl/sites/default/files/2023-07/tno2022_r11961_sdra_status_rapport_2022_-_final_signed_gelakt.pdf) 
are implemented as an optional component. 


- Fragility and consequence models [`SHRA-Groningen-hazard_risk_models`](https://github.com/TNO/SHRA-Groningen-hazard-risk-models) repository)
    - [V5, Crowley & Pinho, 2017](https://nam-onderzoeksrapporten.data-app.nl/reports/download/groningen/en/aaa228dc-71a3-4919-a560-571a4b262a9a); 
    - [V6, Crowley et al., 2019](https://nam-onderzoeksrapporten.data-app.nl/reports/download/groningen/en/85c5dae6-464f-4311-95bc-9b6a21ddf3a8); 
    - [V7, Crowley & Pinho, 2020](https://nam-onderzoeksrapporten.data-app.nl/reports/download/groningen/en/9d8819b7-f0c5-4089-a036-71e755fda328);
    - [TNO 2020](https://open.overheid.nl/documenten/ronl-9884be7a-f42d-495a-b771-22efc7716332/pdf) <br>

- The [`SHRA-Groningen-chaintools`](https://github.com/TNO/SHRA-Groningen-chaintools) repository does not contain models, but generic functionality that is 
used in both the `seismic_source_model` and `hazard_risk_models` code repositories.


## Setting up ##
For the purposes of this document, we will assume that the user has access to a Linux environment. Specifically, the 
code has been tested under _Ubuntu 20.04.6 LTS_. We judge it likely that there will be no or limited problems using 
other Linux-based platforms, WSL (Windows Subsystem for Linux), macOS, or Windows environments. Some additional 
information on this is included under [Setting up virtual Python environment](#setting-up-virtual-python-environments)

### Obtaining the code ####

First, the code needs to be obtained from the relevant code repositories:  
    - [`SHRA-Groningen-seismic_source_model`](https://github.com/TNO/SHRA-Groningen-seismic-source-model) 
    - [`SHRA-Groningen-hazard_risk_models`](https://github.com/TNO/SHRA-Groningen-hazard-risk-models)


  The **recommended** approach is to clone these repositories through [git](https://git-scm.com/) by running:  
  `git clone --recurse-submodules --remote-submodules <repository-address>` <br>
  When using this approach, the `chaintools` repository is automatically included in the correct manner and there is no 
  need to obtain it separately.
  
  Alternatively, a copy of the code can be obtained by using the 'download' buttons of the repositories. In this case, 
  the 'chaintools' folder in `seismic_source_model` and `hazard_risk_models` is downloaded as an empty folder. This 
  folder should be replaced with the full `chaintools` repository which has to be downloaded separately (this means that
  the project should contain the folder `chaintools/chaintools`, the lower of which contains an `__init__.py` file).

### Setting up virtual Python environments ###

To run the code, the user needs to set up Python environments. 
We highly recommend using [mamba](https://github.com/conda-forge/miniforge) or 
[conda](https://docs.conda.io/projects/miniconda/en/latest/) as your package manager, as this ensured that any 
required binaries are taken care of (this is not the case for the default Python package manager pip).
Mamba and Conda also ensure that Python is available on the system, if this was not already the case.

The repositories `seismic_source_model` and `hazard_risk_model` both contain an `environment.yml` file which can be used
to set up a Python environment which contains all the relevant packages (and their correct versions) required to run 
the code as intended. Two virtual environments are required, one for each repository. We do not recommend using a single
Python environment to run the code from both repositories

To set up the virtual environments, the following commands are run (`conda` and `mamba` may be used interchangably): <br>
`mamba env create --f <path_to_seismic_source_model_environment.yml>` <br>
`mamba env create --f <path_to_hazard_risk_models_environment.yml>`

This creates virtual environments with the names 'seismic_source_model' and 'hazard_risk_model'.

At this point, it may be useful to run '`mamba env list`' to obtain the locations where the virtual environment are 
installed on your machine (see [Performing a run](#performing-a-run)).

> The provided `environment.yml` files has references to the exact versions of packages used by the developers, 
which are not available under operating systems other than _Ubuntu 20.04.6 LTS_. In these cases, the less comprehensive 
`environment_light.yml` can be used instead.  However, it should be stressed that this has not been tested extensively 
and may require some custom solutions.

## Obtaining required inputfiles ##

The datafiles that are required as input into the modelchain are published at [Zenodo](https://doi.org/10.5281/zenodo.10245813)

## Performing a run ##
  
For convenience, the calculation of seismicity, seismic hazard, and seismic risk is split up into multiple smaller 
steps. This allows re-use of results that are time-consuming to calculate, and intermediate results can be inspected. <br>

Each step in the modelchain is performed with the same basic command structure: <br>
`<python_executable> <code_for_current_step.py> <configuration_file.yml>` 

where <br> `<python_executable>` is replaced with the path to the appropriate python executable (in one of the two 
virtual environments) <br> 
`<code_for_current_step.py>` is replaced with the path to the python file for that particular step, and <br>
`<configuration_file.yml>` is replaced with the path to the configuration file for that particular step.

>For example, the first step in running the modelchain is parsing the inputfiles to the seismic source model. The code
for that step is `seismic_source_model/parse_input.py`. Since we're running code in `seismic_source_model`, we need to
use the corresponding Python environment. The `python` executable can be called directly (run '`mamba env list`' to 
obtain the locations of the virtual environment), or through `mamba activate`.
An example for a configuration file is provided in the folder `seismic_source_model/example_configs`. The way to run the
first step of the modelchain is then either: <br> <br>
Option 1: <br>
`.../mambaforge/envs/seismic_source_model/bin/python parse_input.py example_configs/parser_config.yml` <br>
Option 2: <br>
`mamba activate seismic_source_model` <br>
`python parse_input.py example_configs/parser_config.yml` <br> <br>
Note that for both the `*.py` files and the configuration files, both full and relative paths can be used.

### Creating the lookup tables and prep-files ###

Being able to re-use previously calculated results also allows us to benefit from working with lookup tables. These 
contain intermediate results that can be pre-calculated, since they do not depend on gas production or source model 
calibration. Specifically, in order to be able to calculate hazard and risk based on a source model, the following 
lookup tables and prep-files need to be pre-calculated: <br>
- rupture prep
- gmm tables (ground motion model tables, required for hazard prep and im_prep)
- fcm tables (fragility and consequence tables, required for risk prep)
- im prep (intensity measure prep, required for risk prep)
- hazard prep
- risk prep
- exposure prep

>`mamba activate hazard_risk_models` <br>
`python rupture_prep.py prep_config.yml` <br>
`python gmm_tables.py prep_config.yml` <br>
`python hazard_prep.py prep_config.yml` <br>
`python fcm_tables.py prep_config.yml` <br>
`python im_prep.py prep_config.yml` <br>
`python risk_prep.py prep_config.yml` <br>
`python exposure_prep.py prep_config.yml` <br>

An example `prep_config.yml` file is provided in the `hazard_risk_models` repository.


### Running the 'backbone' chain ###

Once the lookup tables are available, the calculation of hazard and risk (conditional on a source model, which is in 
turn conditional on a gas pressure scenario) can be performed relatively rapidly:

> The following lines parse the input files required for the source model (earthquake catalogue, pressure grids, 
etc), calibrate the source model on the available earthquake data, and create a forecast.<br>
`mamba activate seismic_source_model` <br>
`python parse_input.py parser_config.yml` <br>
`python calibrate_ssm.py ssm_calibration_config.yml` <br>
`python forecast_ssm.py ssm_forecast_config.yml` <br> <br>
We switch Python environment and then combine the seismicity forecast with the pre-calculated lookup tables to generate
results for hazard and risk.<br>
`mamba activate hazard_risk_models` <br>
`python source_integrator.py source_config.yml` (this step requires a rupture prep to be available)<br>
`python hazard_integrator.py hazard_config.yml` (this step requires a hazard prep to be available) <br> 
`python risk_integrator.py risk_config.yml` (this step requires a risk prep to be available) <br>
`python exposure_integrator.py exposure_config.yml` (this step requires an exposure prep to be available) <br>


### Visualizing the results ###

The calibration and forecast of the seismicity can be visualized with the scripts `visualize_calibration_results.py` and
`visualize_forecast_results.py`. <br> <br>
The hazard and risk results can be visualized in a variety of ways. There are currently no specific tools provided to 
this end. This is foreseen in the upcoming update.



## License ##
Licensed under the [EUPL](/LICENSE)

Copyright (C) 2023 TNO
