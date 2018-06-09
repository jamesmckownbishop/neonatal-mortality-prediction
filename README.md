# Neonatal Mortality Prediction

This project uses public births data to predict neonatal mortality. The scripts are all written for Python 3.6. To replicate the project results, download the scripts to a single folder and run them in the following order on a system with at least 16GB RAM (run times reported on Intel i7-6700 with 16GB RAM):

```Neonatal Mortality Data Preparation.py``` downloads, cleans, and pickles the data; 7.8 minutes

```Neonatal Mortality Data Analysis.py``` runs XGBoost with validation and hyperoptimization using 2016 data; 546 minutes

```Neonatal Mortality Final Test.py``` runs and describes the results of the hyperoptimized models using 2015 data; 4.2 minutes

```Neonatal Mortality Predictor App.py``` predicts mortality given user input; 0.1 minutes

```Neonatal Mortality Predictor List.csv``` is a necessary input file for the latter three scripts that lists the predictors used, their data types, and whether they are included in the prenatal model

```Full 2016 Target Mean Encodings.p``` is a necessary input file for the predictor app that stores the target mean encodings for features treated as categorical

```Prenatal Model.p``` and ```Prenatal Quantiles.p``` are necessary input files for the predictor app that store the prenatal prediction model and its predicted mortality probability quantiles on the training set, respectively

The data analysis and final test scripts require Python packages ```hyperopt``` (which itself requires ```msgpack```) and ```xgboost```; the predictor app script requires ```dash```

## Detailed Instructions

1) Download the Python 3.6 version of Anaconda from https://www.anaconda.com/download/

2) Download this repository to a folder

3) Replace each instance of "~path" in the following with the folder path

```
conda create -n py36_nmp python=3.6.5 -vv
activate py36_nmp

conda install py-xgboost
pip install dash
pip install dash-renderer
pip install dash-html-components
pip install dash-core-components

pip install msgpack
pip install git+git://github.com/hyperopt/hyperopt.git
python "~path\Neonatal Mortality Data Preparation.py"
python "~path\Neonatal Mortality Data Analysis.py"
python "~path\Neonatal Mortality Final Test.py"

python "~path\Neonatal Mortality Predictor App.py"
```

4) Enter the desired commands at a command prompt
- To replicate the analysis, enter all commands (takes about 9.5 hours to complete on Intel i7-6700 with 16GB RAM)
- To skip the analysis and start the app, skip the commands ```pip install msgpack``` through ```python "~path\Neonatal Mortality Final Test.py"```

5) Navigate to http://127.0.0.1:8050/ using your internet browser
