[![Unit tests](https://github.com/CardiacModelling/pcpostprocess/actions/workflows/pytest.yml/badge.svg)](https://github.com//CardiacModelling/pcpostprocess/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/CardiacModelling/pcpostprocess/graph/badge.svg?token=HOL0FrpGqs)](https://codecov.io/gh/CardiacModelling/pcpostprocess)

This repository contains a python package and scripts for handling time-series data from patch-clamp experiments.
The package has been tested with data from a SyncroPatch 384, but may be adapted to work with data in other formats.
It can also be used to perform quality control (QC) as described in [Lei et al. (2019)](https://doi.org/10.1016%2Fj.bpj.2019.07.029).

This package is tested on Ubuntu with Python 3.8, 3.9, 3.10, 3.11, 3.12 and 3.13.

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
python3 -m pip install -e .[test]
```

To run the tests you must first download some test data.
Test data is available at [cardiac.nottingham.ac.uk/syncropatch\_export](https://cardiac.nottingham.ac.uk/syncropatch_export)

```sh
wget https://cardiac.nottingham.ac.uk/syncropatch_export/test_data.tar.xz -P tests/
tar xvf tests/test_data.tar.xz -C tests/
rm tests/test_data.tar.xz
```

Then you can run the tests.
```sh
python3 -m unittest
```

## Usage

### Running QC and post-processing

Quality control (QC) may be run using the criteria outlined in [Rapid Characterization of hERG Channel Kinetics I](https://doi.org/10.1016/j.bpj.2019.07.029) and [Evaluating the predictive accuracy of ion channel models using data from multiple experimental designs](https://doi.org/10.1101/2024.08.16.608289). These criteria assume the use of the `staircase` protocol for quality control, which should be the first and last protocol performed. We also assume the presence of repeats after the addition of an IKr blocker (such as dofetilide).

Prior to performing QC and exporting, an `export_config.py` file should be added to the root of the data directory. This file (see `example_config.py`) contains a Python `dict` (`Q2S_DC`) specifying the filenames of the protocols used for QC, and names they should be outputted with, as well as a Python `dict` (`D2S`) listing the other protocols and names to be used for their output. Additionally, the `saveID` field specifies the name of the expeirment which appears in the output file names.

```sh
$ pcpostprocess run_herg_qc --help

usage: pcpostprocess run_herg_qc [-h] [-c NO_CPUS]
                      [--output_dir OUTPUT_DIR] [-w WELLS [WELLS ...]]
                      [--protocols PROTOCOLS [PROTOCOLS ...]]
                      [--reversal_spread_threshold REVERSAL_SPREAD_THRESHOLD] [--export_failed]
                      [--selection_file SELECTION_FILE] [--subtracted_only]
                      [--figsize FIGSIZE FIGSIZE]
                      [--debug] [--log_level LOG_LEVEL] [--Erev EREV]
                      data_directory

positional arguments:
  data_directory

options:
  -h, --help            show this help message and exit
  -c NO_CPUS, --no_cpus NO_CPUS      Number of workers to spawn in the multiprocessing pool (default: 1)
  --output_dir OUTPUT_DIR      path where the output will be saved
  -w WELLS [WELLS ...], --wells WELLS [WELLS ...]   wells to include (default: all)
  --protocols PROTOCOLS [PROTOCOLS ...]  protocols to include (default: all)
  --reversal_spread_threshold REVERSAL_SPREAD_THRESHOLD       The maximum spread in reversal potential (across sweeps) allowed for QC
  --export_failed                Flag specifying whether to produce full output for those wells failing QC (default: false)
  --selection_file SELECTION_FILE      File listing wells to be included
  --figsize FIGSIZE FIGSIZE
  --debug
  --log_level LOG_LEVEL
  --Erev EREV           The reversal potential during the experiment
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

