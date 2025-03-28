## General modelchain documentation ##
<br>

See [CHAIN MANUAL](/CHAIN_MANUAL.md)

## Seismic source model documentation ##

 The **recommended** approach is to clone this repository through [git](https://git-scm.com/) by running:  
  `git clone --recurse-submodules --remote-submodules <repository-address>` <br>
  When using this approach, the `chaintools` repository is automatically included in the correct manner and there is no  need to obtain it separately.
  
  Alternatively, a copy of the code can be obtained by using the 'download' button. 
  In this case, the 'chaintools' folder is downloaded as an empty folder. This folder should be 
  replaced with the full [`chaintools`](https://github.com/TNO/SHRA-Groningen-chaintools) repository which has to be downloaded separately (this means that
  the project should contain the folder `chaintools/chaintools`, the lower of which contains an `__init__.py` file).

### Setting up virtual Python environments ###

To run the code, the user needs to set up a Python environment. 
We highly recommend using [mamba](https://github.com/conda-forge/miniforge) or 
[conda](https://docs.conda.io/projects/miniconda/en/latest/) as your package manager, as this ensured that any 
required binaries are taken care of (this is not the case for the default Python package manager pip).
Mamba and Conda also ensure that Python is available on the system, if this was not already the case.

The repository contains an `environment.yml` file which can be used to set up a Python environment which contains all the relevant packages (and their correct versions) required to run 
the code as intended. 

To set up the virtual environment, the following command is run (`conda` and `mamba` may be used interchangably): <br>
`mamba env create -f <path_to_environment.yml>` <br>

> The provided `environment.yml` files has references to the exact versions of packages used by the developers, 
which are not available under operating systems other than _Ubuntu 20.04.6 LTS_. In these cases, the less comprehensive 
`environment_light.yml` can be used instead.  However, it should be stressed that this has not been tested extensively 
and may require some custom solutions.

This creates a virtual environment with the name 'seismic_source_model'.

### Obtaining required inputfiles ###

The datafiles that are required as input into the modelchain are published at [Zenodo](https://doi.org/10.5281/zenodo.10245813)

### Running the seismic source model ###


`mamba activate seismic_source_model` <br>
`python parse_input.py parser_config.yml` <br>
`python calibrate_ssm.py ssm_calibration_config.yml` <br>
`python forecast_ssm.py ssm_forecast_config.yml` <br> <br>


Example `config.yml` files are provided in [example_configs](/example_configs)

## License ##
Licensed under the [EUPL](/LICENSE)

Copyright (C) 2023 TNO
