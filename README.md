# pv_output_prediction
This is a Machine Learning project to train a model on PV output data and global irradiance in order to predict output 
of solar panels in other data from other time series. The purpose of this project is to test and demonstrate my 
knowledge of Machine learning. I would be happy for any input or corrections on my methodology (both ML and the coding). 

It optimizes several models from SciKit-Learn (see below) using SciKit-Optimize 'BayesSearchCV', including trying 
voting regressors and stacking regressors. The best model is then selected and saved to be used for predictions.

| Regressor models          |
|---------------------------|
| LinearSVR                 |
| SGDRegressor              |
| Ridge                     |
| Lasso                     |
| ElasticNet                |
| LinearRegression          |
| DecisionTreeRegressor     |
| RandomForestRegressor     |
| ExtraTreesRegressor       |
| AdaBoostRegressor         |
| GradientBoostingRegressor |
| VotingRegressor           |
| StackingRegressor         |

## trained model
The best model, GradientBoostingRegressor, have a cross-validation score of -XX.XXX kWh negative mean absolute error 
(MAE). 

The choice of negative MAE as a score was chosen because the global irradiance dataset is right skewed (Skewness: 1.84), 
heavy-tailed (Kurtosis: 2.23) and has relatively many outliers (1.88% of samples), which MAE is more robust against.
See the outlier and errors analysis in the Jupyter Notebook in the folder 'data_analysis'.

This model is fitted to the full dataset and output as a .joblib file to be used for predictions for the AI 
Solar Battery Arbitrage project. The model is trained using a seed of 19475.

### Predictions
These can be found in the folder 'predictions' and the original data from SMHI can be found in 'data'.
