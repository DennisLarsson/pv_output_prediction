# Photovoltaic output modeling and prediction using SciKit machine learning
This is a Machine Learning project to train a model on PV output data and global irradiance in order to predict output 
of solar panels in other data from other time series. The purpose of this project is to test and demonstrate my 
knowledge of Machine learning. I would be happy for any input or corrections on my methodology (both ML and the coding). 

## Model optimization
It optimizes several models from SciKit-Learn (see below) using SciKit-Optimize 'BayesSearchCV', including trying 
voting regressors and stacking regressors. The best model is then selected and saved to be used for predictions.

| Regressor models                                   | Negative MAE |
|----------------------------------------------------|--------------|
| LinearSVR                                          | -5.94        |
| SGDRegressor                                       | -6.42        |
| Ridge                                              | -6.49        |
| Lasso                                              | -6.28        |
| ElasticNet                                         | -6.49        |
| LinearRegression                                   | -6.50        |
| DecisionTreeRegressor                              | -2.58        |
| RandomForestRegressor                              | -1.95        |
| ExtraTreesRegressor                                | -1.76        |
| AdaBoostRegressor                                  | -8.11        |
| GradientBoostingRegressor                          | -1.43        |
| __Ensemble models__                                | ---          |
| VotingRegressor (wo/ weights)                      | -3.96        |
| VotingRegressor (with weights)                     | -2.10        |
| StackingRegressor (SVM, GBR, ETR, Lasso, FE*: RFR) | -1.63        |
| StackingRegresso (All optimized models, FE*: Ridge | -1.55        |
*FE = Final Estimator

### Input data
The input data for training is simulated hourly PVGIS data from https://re.jrc.ec.europa.eu/pvg_tools/en/#HR for 2022 
to 2023 with coordinates from outside of Stockholm, Sweden. The setup was a 1kW peak crystalline silicon cells with 
default 14% system loss and mounted with an slope of 42° and 0° azimuth.

The data contains time and date of year, PV output and G(i) (global irradiance). The time and day of year is converted 
to sine and cosine to better capture their cyclical nature.

The G(i) and sine and cosine time and day are used as training data and the PV output as used as labels.

### trained model
The script saves the scores of the best estimator for each model in a file 'model_results.joblib' called in the 
'trained_fitted_model' folder.

The best model, GradientBoostingRegressor, have a mean cross-validation score of -1.43 W (Watts) negative mean absolute 
error (MAE). Given that the dataset has a range of values from 0 to 905.79 W, a mean of 118.32 W and a standard 
deviation of 212.28, a mean absolute error of around 1.43 W is 1.21% of the mean value and thus fairly accurate.

The choice of negative MAE as a score was chosen because the global irradiance dataset is right skewed (Skewness: 1.84), 
heavy-tailed (Kurtosis: 2.23) and has relatively many outliers (1.88% of samples), which MAE is more robust against.
See the outlier and errors analysis in the Jupyter Notebook in the folder 'data_analysis'.

This model is fitted to the full dataset and output as a .joblib file to be used for predictions for the AI 
Solar Battery Arbitrage project. The model is trained using a seed of 19475.

## Predictions
Predictions of PV output were made using global irradiance data from SMHI https://strang.smhi.se/extraction/index.php. 
Data was collected from both January 1st 2025 to December 31st 2025 and from January 1st 2026 to May 31st 2026 and can 
be found in the folder 'data'.

training data in the form of global irradiance and sine and cosine time and day was used as input for the above fitted 
model to predict PV output for both the 2025 and 2026 dataset.

The results of this can be found in the folder 'predictions'.
