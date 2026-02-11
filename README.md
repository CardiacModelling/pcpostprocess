[![Unit tests](https://github.com/CardiacModelling/pcpostprocess/actions/workflows/tests.yml/badge.svg)](https://github.com//CardiacModelling/pcpostprocess/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/CardiacModelling/pcpostprocess/graph/badge.svg?token=HOL0FrpGqs)](https://codecov.io/gh/CardiacModelling/pcpostprocess)

This repository contains a python package and scripts for handling time-series data from patch-clamp experiments.
The package has been tested with data from a SyncroPatch 384, but may be adapted to work with data in other formats.
It can also be used to perform quality control (QC) as described in [Lei et al. (2019)](https://doi.org/10.1016%2Fj.bpj.2019.07.029).

This package is tested on Ubuntu with Python 3.10 to 3.14.

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
python3 -m pip install -e .'[dev]'
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


## Contributing

Although `pcpostprocess` is still early in its development, we have set up guidelines for user contributions in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Licensing

For licensing information, see the [LICENSE](./LICENSE) file.

