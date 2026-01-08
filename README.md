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

We are working to make pcpostprocess a reusable tool for different types of data.
At the moment, it is geared towards the specific use case detailed below.

### Experimental data

- Syncropatch 384 ("DataControl 384") data
- The full set of protocols is repeated after addition of a strong IKr blocker, e.g. E-4031
- Each set of protocols starts and ends with a "staircase" protocol, leading to a total of 4 staircases:
  - Before and after all other protocols, before drug block
  - Before and after all other protocols, after drug block

So a full set of experiments would go "staircase, other1, other2, ... staircase, drug addition & wash in, staircase, other1, other2, ..., staircase".

Each protocol
- Starts with a leak estimation ramp
- Followed by a step to 40mV and then to -120mV
- Ends with a reversal potential step
- Provides an estimate of Rseal, Rseries, and Cm

### Leak correction

- Leak correction is applied by estimating linear leak from the leak step.
- Before and after drug traces are both corrected
- The final signal is leak-corrected-before minus leak-corrected-after

### Quality control

Quality control is based on [Lei et al. (2019) Rapid Characterization of hERG Channel Kinetics I](https://doi.org/10.1016/j.bpj.2019.07.029) and [Shuttleworth et al. (2025) Evaluating the predictive accuracy of ion channel models using data from multiple experimental designs](https://doi.org/10.1098/rsta.2024.0211).

First, all four staircase protocols are checked in all wells using the Lei et al. criteria, along with new measures described in Shuttleworth et al.
Any wells not rejected are then tested using a smaller set of (less protocol specific) criteria detailed in Shuttleworth et al.
Only wells passing all tests on all protocols are retained.

### Writing an export_config.py

Prior to performing QC and exporting, an `export_config.py` file should be added to the root of the data directory.
An example file is available [here](./example_config.py).
This should contain three entries:

1. A `saveID` variable, providing a name to use in exported data. 
2. A dictionary variable `Q2S_DC` indicating the name of the staircase protocol, and mapping it onto an more user-friendly name used in export.
3. A dictionary variable `D2S` indicating names of other protocols to export, again mapping onto names for export.

For example, in the test data (see "Getting Started") above, we have six subdirectories:
- `staircaseramp (2)_2kHz_15.01.07`, staircase run as first protocol before E-4031
- `StaircaseInStaircaseramp (2)_2kHz_15.01.51`, non-QC protocol
- `staircaseramp (2)_2kHz_15.06.53`, staircase run as last protocol before E-4031
- `staircaseramp (2)_2kHz_15.11.33`, staircase run as first protocol after E-4031 addition
- `StaircaseInStaircaseramp (2)_2kHz_15.12.17`, non-QC protocol
- `staircaseramp (2)_2kHz_15.17.19`, staircase run as final protocol

Here each directory name is a protocol name followed by a timestamp.
Corresponding dictionaries could be `Q2S_DC = {'staircaseramp (2)_2kHz': 'staircase'}` (indicating the staircase protocol) and `D2S = {'StaircaseInStaircaseramp (2)_2kHz': 'sis'}`.
The saveID could be any string, but in our example we use `saveID = '13112023_MW2'`

### Running

To run, we call `run_herg_qc` with:
```sh
pcpostprocess run_herg_qc test_data/13112023_MW2_FF -o output --output_traces -w A01 A02 A03
```
Here:
- `test_data/13112023_MW2_FF` is the path to the test data,
- `-o output` tells the script to store all results in the folder `output`
- `--output_traces` tells the script to export the data (instead of running QC only), and

The last part, `-w A01 A02 A03`, tells the script to only check the first three wells.
This speeds things up for testing, but would be omitted in normal use.

To see the full set of options, use 

```sh
$ pcpostprocess run_herg_qc --help
```

### Interpreting the output

After running the command above on the test data provided, the directory `output` will contain the subdirectories:

- `-120mV time constant` Plots illustrating the fits performed when estimating time constants. These are output but not used in QC.
- `leak_correction` Plots illustrating the _linear_ leak correction process in all wells passing staircase QC.
- `qc3-bookend` Plots illustrating the "QC3 bookend" criterion used in Shuttleworth et al.
- `reversal_plots` Plots illustrating the reversal potential estimation
- `subtraction_plots` Plots illustrating the drug subtracted trace leak removal.
- `traces` The exported traces (times, voltages, currents)

and the files:

- `chrono.txt` Lists the order in which protocols were run. For the staircase, the repeat at the end of the sequence is indicated by an added `_2`. 
- `passed_wells.txt` Lists the wells that **passed all QC**.
- `pcpostprocess_info.txt` Information on the run that generated this output (date, time, command used etc.)
- `QC-13112023_MW2.csv` A CSV representation of the pass/fail results for each individual criterion
- `QC-13112023_MW2.json` A JSON representation of the same
- `qc_table.tex` A table in Tex format with the above.
- `qc_vals_df.csv` Numerical values used in the final QC.
- `selected-13112023_MW2.txt` **Preliminary QC selection** based on staircase protocol 1 (before other protocols) and 2 (after other protocols)
- `selected-13112023_MW2-staircaseramp.txt` Preliminary selection based on staircase 1
- `selected-13112023_MW2-staircaseramp_2.txt` Preliminary selection based on staircase 2
- `subtraction_qc.csv` Numerical values used in the final QC and leak subtraction.

## Contributing

Although `pcpostprocess` is still early in its development, we have set up guidelines for user contributions in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Licensing

For licensing information, see the [LICENSE](./LICENSE) file.

