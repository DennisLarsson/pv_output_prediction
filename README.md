# pv_output_prediction
This is a Machine Learning project to train a model on PV output data and global irradiance in order to predict output 
of solar panels in other data from other time series.

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
(MAE). This model is fitted to the full dataset and output as a .joblib file to be used for predictions for the AI 
Solar Battery Arbitrage project. The model is trained using a seed of 19475.

### Predictions
These can be found in the folder 'predictions' and the original data from SMHI can be found in 'data'.
