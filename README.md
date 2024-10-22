
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Unit tests](https://github.com/CardiacModelling/pcpostprocess/actions/workflows/pytest.yml/badge.svg)](https://github.com//CardiacModelling/pcpostprocess/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/CardiacModelling/pcpostprocess/graph/badge.svg?token=HOL0FrpGqs)](https://codecov.io/gh/CardiacModelling/pcpostprocess)


<!-- PROJECT LOGO -->
<!-- <br /> -->
<!-- <div align="center"> -->
<!--     <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
<!--   </a> -->


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project
This project contains a python package and scripts for handling time-series data from patch-clamp experiments. The package has been tested with data from a SyncroPatch 384, but may be adapted to work with data in other formats. The package can also be used to perform quality control (QC) as described in [Lei et al. (2019)](https://doi.org/10.1016%2Fj.bpj.2019.07.029).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This package has been tested on Ubuntu with Python 3.7, 3.8, 3.9, 3.10 and 3.11.

### Installation

First clone this repository

```
git clone git@github.com:CardiacModelling/pcpostprocess && cd pcpostprocess
```

With one of these versions install, create and activate a virtual environment.

  ```sh
  python3 -m venv .venv && source .venv/bin/activate
  ```

Then install the package with `pip`.

```
python3 -m pip install --upgrade pip && python3 -m pip install -e .'[test]'
```

To run the tests you must first download some test data. Test data is available at [cardiac.nottingham.ac.uk/syncropatch\_export](https://cardiac.nottingham.ac.uk/syncropatch_export)

```
wget https://cardiac.nottingham.ac.uk/syncropatch_export/test_data.tar.xz -P tests/
tar xvf tests/test_data.tar.xz -C tests/
```

Then you can run the tests.
```
python3 -m unittest
```


<!-- USAGE -->
## Usage

### Running QC and post-processing

Quality control (QC) may be run using the criteria outlined in [Rapid Characterization of hERG Channel Kinetics I](https://doi.org/10.1016/j.bpj.2019.07.030) and [Evaluating the predictive accuracy of ion channel models using data from multiple experimental designs](https://doi.org/10.1101/2024.08.16.608289). These criteria assume the use of the `staircase` protocol for quality control, which should be the first and last protocol performed. We also assume the presence of repeats after the addition of an IKr blocker (such as dofetilide).

Prior to performing QC and exporting, an `export*config.py` file should be added to the root of the data directory. This file (see `example*config.py`) contains a Python `dict` (`Q2S_DC`) specifying the filenames of the protocols used for QC, and names they should be outputted with, as well as `dict` (`D2S`) listing the other protocols and names to be used for their output. Additionally, the `saveID` field specifies the name of the expeirment which is used throughout the output.

```
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
  -c NO_CPUS, --no_cpus NO_CPUS
  --output_dir OUTPUT_DIR
  -w WELLS [WELLS ...], --wells WELLS [WELLS ...]
  --protocols PROTOCOLS [PROTOCOLS ...]
  --reversal_spread_threshold REVERSAL_SPREAD_THRESHOLD
  --export_failed
  --selection_file SELECTION_FILE
  --subtracted_only
  --figsize FIGSIZE FIGSIZE
  --debug
  --log_level LOG_LEVEL
  --Erev EREV
```


### Exporting Summary

```
$ pcpostprocess summarise_herg_export --help

usage: pcpostprocess summarise_herg_export [-h] [--cpus CPUS]
                      [--wells WELLS [WELLS ...]] [--output OUTPUT]
                      [--protocols PROTOCOLS [PROTOCOLS ...]] [-r REVERSAL]
                      [--experiment_name EXPERIMENT_NAME]
                      [--figsize FIGSIZE FIGSIZE] [--output_all]
                      [--log_level LOG_LEVEL]
                      data_dir qc_estimates_file

positional arguments:
  data_dir           path to the directory containing the subtract_leak results
  qc_estimates_file

options:
  -h, --help            show this help message and exit
  --cpus CPUS, -c CPUS
  --wells WELLS [WELLS ...], -w WELLS [WELLS ...]
  --output OUTPUT, -o OUTPUT
  --protocols PROTOCOLS [PROTOCOLS ...]
  -r REVERSAL, --reversal REVERSAL
  --experiment_name EXPERIMENT_NAME
  --figsize FIGSIZE FIGSIZE
  --output_all
  --log_level LOG_LEVEL
```


<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Joseph Shuttleworth joseph.shuttleworth@nottingham.ac.uk

Project Link: [https://github.com/CardiacModelling/pcpostprocess](https://github.com/CardiacModelling/pcpostprocess)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/CardiacModelling/pcpostprocess.svg?style=for-the-badge
[contributors-url]: https://github.com/CardiacModelling/pcpostprocess/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/CardiacModelling/pcpostprocess.svg?style=for-the-badge
[forks-url]: https://github.com/CardiacModelling/pcpostprocess/network/members
[stars-shield]: https://img.shields.io/github/stars/CardiacModelling/pcpostprocess.svg?style=for-the-badge
[stars-url]: https://github.com/CardiacModelling/pcpostprocess/stargazers
[issues-shield]: https://img.shields.io/github/issues/CardiacModelling/pcpostprocess.svg?style=for-the-badge
[issues-url]: https://github.com/CardiacModelling/pcpostprocess/issues
[license-shield]: https://img.shields.io/github/license/Cardiac/Modelling/pcpostprocess.svg?style=for-the-badge
[license-url]: https://github.com/CardiacModelling/pcpostprocess/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
