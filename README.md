# Time-series forecasting with Machine Learning

This folder contains notebooks to perform perliminary data analysis and time-series forecasting with machine learning using Python.

## Installation

To work with the repository, one can simply clone it in the local machine:

```bash
git clone "https://github.com/andreabellome/time_series_forecasting"
```

The toolbox uses common Python libraries: [numpy](https://numpy.org/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/) and [plotly](https://plotly.com/python/). If not already downloaded, please use [pip](https://pip.pypa.io/en/stable/) to do so:

```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install plotly
```

Moreover, machine-learning-specific libraries are required: [lightgbm](https://pypi.org/project/lightgbm/) for the regression models and [skforecast](https://skforecast.org/0.11.0/index.html) for the Bayesian search. These can be installed using [pip](https://pip.pypa.io/en/stable/) as well:

```bash
pip install lightgbm
pip install skforecast
```

## License

The work is under license [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc/4.0/), that is an Attribution Non-Commercial license. One can find the specifics in the [LICENSE](/LICENSE) file.

Only invited developers can contribute to the repository.

## Usage

A notebook is provided that is [notebook_power_load_forecasting.ipynb](/notebook_power_load_forecasting.ipynb). This is self-explanatory, and the interested developer is encouraged to explore the code.












