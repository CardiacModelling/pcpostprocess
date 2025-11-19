[![Unit tests](https://github.com/CardiacModelling/pcpostprocess/actions/workflows/tests.yml/badge.svg)](https://github.com//CardiacModelling/pcpostprocess/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/CardiacModelling/pcpostprocess/graph/badge.svg?token=HOL0FrpGqs)](https://codecov.io/gh/CardiacModelling/pcpostprocess)

This repository contains a python package and scripts for handling time-series data from patch-clamp experiments.
The package has been tested with data from a SyncroPatch 384, but may be adapted to work with data in other formats.
It can also be used to perform quality control (QC) as described in [Lei et al. (2019)](https://doi.org/10.1016%2Fj.bpj.2019.07.029).

This package is tested on Ubuntu with Python 3.10, 3.11, 3.12, 3.13 and 3.14.

## Getting Started

First clone the repository

```sh
git clone git@github.com:CardiacModelling/pcpostprocess
cd pcpostprocess
```

Create and activate a virtual environment.

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Then install the package with `pip`.

```sh
python3 -m pip install --upgrade pip
python3 -m pip install -e .'[test]'
```

To run the tests you must first download some test data.
Test data is available at [cardiac.nottingham.ac.uk/syncropatch\_export](https://cardiac.nottingham.ac.uk/syncropatch_export)

```sh
wget https://cardiac.nottingham.ac.uk/syncropatch_export/test_data.tar.xz
tar xvf test_data.tar.xz
rm test_data.tar.xz
```

Then you can run the tests.
```sh
python3 -m unittest
```

## Usage

### Classic case

- Syncropatch 384 ("DataControl 384" output)
- A full repeat of all protocols with an IKr blocker, e.g. E-4031
- A staircase protocol, run 4 times:
  - Before and after all other protocols, before drug block
  - Before and after all other protocols, after drug block
- Additional protocols ran in between the staircases
- Leak subtraction by...


Quality control (QC) may be run using the criteria outlined in [Rapid Characterization of hERG Channel Kinetics I](https://doi.org/10.1016/j.bpj.2019.07.029) and [Evaluating the predictive accuracy of ion channel models using data from multiple experimental designs](https://doi.org/10.1101/2024.08.16.608289). 
These criteria assume the use of the `staircase` protocol for quality control, which should be the first and last protocol performed. 
We also assume the presence of repeats after the addition of an IKr blocker (such as dofetilide).

Prior to performing QC and exporting, an `export_config.py` file should be added to the root of the data directory. 
This file (see `example_config.py`) contains a Python `dict` (`Q2S_DC`) specifying the filenames of the protocols used for QC, and names they should be outputted with, as well as a Python dict (`D2S`) listing the other protocols and names to be used for their output. 
Additionally, the `saveID` field specifies the name of the expeirment which appears in the output file names.

An example input directory might contain the following subdirectories
- `staircaseramp (2)_2kHz_15.01.07`, staircase 1 before E-4031
- `StaircaseInStaircaseramp (2)_2kHz_15.01.51`, non-QC protocol
- `staircaseramp (2)_2kHz_15.06.53`, staircase 2 before E-4031
- `staircaseramp (2)_2kHz_15.11.33`, staircase 1 after E-4031
- `StaircaseInStaircaseramp (2)_2kHz_15.12.17`, non-QC protocol
- `staircaseramp (2)_2kHz_15.17.19`, staircase 2 after E-4031

`run_herg_qc` will use the time parts of the staircase names to work out which is which.
This assumes there are either 2 staircase runs (once before drug, once after), or 4 (as above)

Example:
```
# Configuration for run_herg_qc

# Save name for this set of data
saveID = 'EXPERIMENT_NAME'

# Name of protocols to use in QC, mapped onto names used in output
D2S_QC = {
    'staircaseramp (2)_2kHz': 'staircaseramp'
}

# Name of additional protocols to export, but not use in QC (again as a dict)
D2S = {
    'StaircaseInStaircaseramp (2)_2kHz': 'sis',
}
```

Next, we call `run_herg_qc`:
```sh
pcpostprocess run_herg_qc test_data/13112023_MW2_FF -w A01 A02 A03 -o output --output_traces

```



```sh
$ pcpostprocess run_herg_qc --help
```






### Exporting Summary

The `summarise_herg_export` command produces additionally output after `run_herg_qc` has been run.

```sh
$ pcpostprocess summarise_herg_export --help

usage: pcpostprocess summarise_herg_export [-h] [--cpus CPUS]
                      [--wells WELLS [WELLS ...]] [--output OUTPUT]
                      [--protocols PROTOCOLS [PROTOCOLS ...]] [-r REVERSAL]
                      [--experiment_name EXPERIMENT_NAME]
                      [--figsize FIGSIZE FIGSIZE] [--output_all]
                      [--log_level LOG_LEVEL]
                      data_dir qc_estimates_file

positional arguments:
  data_dir           path to the directory containing the run_herg_qc results

options:
  -h, --help            show this help message and exit
  --cpus CPUS, -c CPUS
  --wells WELLS [WELLS ...], -w WELLS [WELLS ...]   wells to include in the output (default: all)
  --output OUTPUT, -o OUTPUT     path where the output will be saved
  --protocols PROTOCOLS [PROTOCOLS ...]  protocols to include (default: all)
  -r REVERSAL, --reversal REVERSAL   the reversal potential during the experiment
  --experiment_name EXPERIMENT_NAME
  --figsize FIGSIZE FIGSIZE
  --output_all        Flag specifying whether to output all plots (default: false)
  --log_level LOG_LEVEL
```

## Contributing

Although `pcpostprocess` is still early in its development, we have set up guidelines for user contributions in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Licensing

For licensing information, see the [LICENSE](./LICENSE) file.

