# Neonatal Mortality Prediction

This project uses public births data to predict neonatal mortality. The scripts are all written for Python 3. Download the scripts to a single folder and run them in the following order on a system with at least 16GB RAM (times reported on Intel i7-6700 with 16GB RAM):

Neonatal Mortality Data Preparation - downloads, cleans, and pickles the data; 6 minutes

Neonatal Mortality Data Analysis - runs XGBoost with validation and hyperoptimization using 2016 data; 12.7 hours

Neonatal Mortality Final Test - runs and describes the results of the hyperoptimized models using 2015 data; 20 minutes

Neonatal Mortality Predictor App - predicts mortality given user input

The Neonatal Mortality Predictor List is a necessary input file for each script that lists the predictors used, their data types, and whether they are known prior to birth

The data analysis and final test scripts require hyperopt, sklearn, and xgboost

The predictor app script requires dash
