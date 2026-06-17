from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, VotingRegressor, StackingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np
import joblib
from process_pvgis import process_pvgis

def get_search_spaces():
    search_spaces = {
        'LinearRegression': {
            'fit_intercept': Categorical([True, False]),
        },

        'DecisionTreeRegressor': {
            'max_depth': Integer(3, 30, 'uniform'),
            'min_samples_split': Integer(2, 50, 'uniform'),
            'min_samples_leaf': Integer(1, 25, 'uniform'),
            'criterion': Categorical(['absolute_error', 'squared_error', 'poisson']),
        },

        'RandomForestRegressor': {
            'n_estimators': Integer(100, 1000, 'uniform'),
            'max_depth': Integer(3, 30, 'uniform'),
            'min_samples_split': Integer(2, 50, 'uniform'),
            'min_samples_leaf': Integer(1, 25, 'uniform'),
            'bootstrap': Categorical([True, False]),
        },

        'LinearSVR': {
            'model__C': Real(0.01, 1000, 'log-uniform'),
            'model__epsilon': Real(0.001, 5, 'log-uniform'),
            'model__loss': Categorical(['epsilon_insensitive', 'squared_epsilon_insensitive']),
            'model__max_iter': Integer(10000, 100000, 'uniform'),
        },

        'SGDRegressor': {
            'model__alpha': Real(1e-6, 1, 'log-uniform'),
            'model__max_iter': Integer(100, 2000, 'uniform'),
            'model__tol': Real(1e-5, 1e-2, 'log-uniform'),
            'model__penalty': Categorical(['l2', 'l1', 'elasticnet']),
            'model__learning_rate': Categorical(['invscaling', 'constant', 'adaptive']),
        },

        'Ridge': {
            'model__alpha': Real(1e-4, 100, 'log-uniform'),
            'model__fit_intercept': Categorical([True, False]),
            'model__solver': Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
        },

        'Lasso': {
            'model__alpha': Real(1e-4, 100, 'log-uniform'),
            'model__fit_intercept': Categorical([True, False]),
            'model__selection': Categorical(['cyclic', 'random']),
        },

        'ElasticNet': {
            'model__alpha': Real(1e-4, 100, 'log-uniform'),
            'model__l1_ratio': Real(0.01, 1.0, 'uniform'),
            'model__fit_intercept': Categorical([True, False]),
            'model__selection': Categorical(['cyclic', 'random']),
        },

        'ExtraTreesRegressor': {
            'n_estimators': Integer(100, 1000, 'uniform'),
            'max_depth': Integer(3, 30, 'uniform'),
            'min_samples_split': Integer(2, 50, 'uniform'),
            'min_samples_leaf': Integer(1, 25, 'uniform'),
            'bootstrap': Categorical([True, False]),
        },

        'AdaBoostRegressor': {
            'n_estimators': Integer(100, 1000, 'uniform'),
            'learning_rate': Real(0.001, 0.5, 'log-uniform'),
            'loss': Categorical(['linear', 'square', 'exponential']),
        },

        'GradientBoostingRegressor': {
            'n_estimators': Integer(100, 1000, 'uniform'),
            'learning_rate': Real(0.001, 0.5, 'log-uniform'),
            'max_depth': Integer(3, 15, 'uniform'),
            'min_samples_split': Integer(2, 50, 'uniform'),
            'min_samples_leaf': Integer(1, 25, 'uniform'),
            'subsample': Real(0.5, 1.0, 'uniform'),
        },
    }
    return search_spaces

def create_models():
    print("Creating models...")
    scaled_models = {
        'LinearSVR': (Pipeline([('scaler', StandardScaler()), ('model', LinearSVR())]),
                      get_search_spaces()['LinearSVR']),
        'SGDRegressor': (Pipeline([('scaler', StandardScaler()), ('model', SGDRegressor())]),
                         get_search_spaces()['SGDRegressor']),
        'Ridge': (Pipeline([('scaler', StandardScaler()), ('model', Ridge())]), get_search_spaces()['Ridge']),
        'Lasso': (Pipeline([('scaler', StandardScaler()), ('model', Lasso())]), get_search_spaces()['Lasso']),
        'ElasticNet': (Pipeline([('scaler', StandardScaler()), ('model', ElasticNet())]),
                       get_search_spaces()['ElasticNet']),
    }

    unscaled_models = {
        'LinearRegression': (LinearRegression(), get_search_spaces()['LinearRegression']),
        'DecisionTreeRegressor': (DecisionTreeRegressor(), get_search_spaces()['DecisionTreeRegressor']),
        'RandomForestRegressor': (RandomForestRegressor(), get_search_spaces()['RandomForestRegressor']),
        'ExtraTreesRegressor': (ExtraTreesRegressor(), get_search_spaces()['ExtraTreesRegressor']),
        'AdaBoostRegressor': (AdaBoostRegressor(), get_search_spaces()['AdaBoostRegressor']),
        'GradientBoostingRegressor': (GradientBoostingRegressor(), get_search_spaces()['GradientBoostingRegressor']),
    }

    models = {**scaled_models, **unscaled_models}
    return models

def run_bayes_search(model_set, n_iter=50):
    print("Running Bayes Search...")
    results = {}
    for name, (model, search_space) in model_set.items():
        if not search_space:
            results[name] = {'model': model, 'best_params': {}, 'best_score': None}
            continue

        opt = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
        )

        opt.fit(X_train, y_train.values.ravel())

        results[name] = {
            'model': opt,
            'best_params': opt.best_params_,
            'best_score': opt.best_score_,
        }

    for name, result in results.items():
        print(f"{name}:")
        print(f"  Best parameters: {result['best_params']}")
        print(f"  Best score: {result['best_score']}")
        print()

    return results

def get_best_models(model_results):
    best_models = {}
    for name, result in model_results.items():
        if result['best_params']:
            best_model = clone(result['model'].best_estimator_)
            best_models[name] = best_model
        else:
            best_models[name] = result['model']

    return best_models

#def get_individual_scores(best_models):
#    individual_scores = {}
#    for name, model in best_models.items():
#        scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=3, scoring="neg_mean_absolute_error")
#        individual_scores[name] = np.mean(scores)
#
#    print("\nIndividual Model Mean CV Scores:")
#    for name, score in individual_scores.items():
#        print(f"{name}: {score}")
#
#    return individual_scores

def get_weights(best_results):
    inv_scores = {name: 1 / (1 + score['best_score']) for name, score in best_results.items()}
    total = sum(inv_scores.values())
    weights = [inv_scores[name] / total for name in best_models]

    return weights

def create_voting_regressor(models, weights=None):
    print("Creating voting regressor...")
    voting_regressor = VotingRegressor(
        estimators=[(name, model) for name, model in models.items()],
        n_jobs=-1,
        weights=weights,
    )

    return voting_regressor

def create_stacking_regressor(models, all_model=True, final_estimator='Ridge'):
    print("Creating stacking regressor...")
    if all_model:
        stacking_regressor = StackingRegressor(
            estimators=[(name, model) for name, model in best_models.items()],
            final_estimator=clone(best_models[final_estimator]),
            cv=3,
            n_jobs=-1
        )
    else:
        base_models = [
            ('LinearSVR', best_models['LinearSVR']),
            ('GradientBoostingRegressor', best_models['GradientBoostingRegressor']),
            ('ExtraTreesRegressor', best_models['ExtraTreesRegressor']),
            ('Lasso', best_models['Lasso']),
        ]

        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=clone(best_models[final_estimator]),
            cv=3,
            n_jobs=-1
        )

    return stacking_regressor


def print_ensemble_results(voting_scores, voting_weights_scores, stacking_scores, stacking_all_scores):
    print("Voting Regressor CV Scores:", voting_scores)
    print("Mean CV Score:", np.mean(voting_scores))
    print()
    print("Voting Regressor weights CV Scores:", voting_weights_scores)
    print("Mean CV Score:", np.mean(voting_weights_scores))
    print()
    print("Stacking Regressor CV Scores:", stacking_scores)
    print("Mean CV Score:", np.mean(stacking_scores))
    print()
    print("Stacking Regressor all CV Scores:", stacking_all_scores)
    print("Mean CV Score:", np.mean(stacking_all_scores))

if __name__ == "__main__":

    PVGIS_filename = "data/Timeseries_59.246_18.035_SA3_1kWp_crystSi_14_42deg_0deg_2022_2023.csv"

    X_train, X_test, y_train, y_test = process_pvgis(PVGIS_filename)

    models = create_models()
    best_results = run_bayes_search(models, n_iter = 50)
    best_models = get_best_models(best_results)

    #individual_scores = get_individual_scores(best_models)
    weights = get_weights(best_results)

    voting_regressor = create_voting_regressor(best_models)
    voting_scores = cross_val_score(
        voting_regressor,
        X_train,
        y_train.values.ravel(),
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    voting_regressor_weights = create_voting_regressor(best_models, weights)
    voting_weights_scores = cross_val_score(
        voting_regressor_weights,
        X_train,
        y_train.values.ravel(),
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    stacking_regressor = create_stacking_regressor(models, all_model=False, final_estimator='RandomForestRegressor')
    stacking_scores = cross_val_score(
        stacking_regressor,
        X_train,
        y_train.values.ravel(),
        cv=3,
        scoring="neg_mean_absolute_error"
        )

    stacking_regressor_all = create_stacking_regressor(models, all_model=True, final_estimator='Ridge')
    stacking_all_scores = cross_val_score(
        stacking_regressor_all,
        X_train,
        y_train.values.ravel(),
        cv=3,
        scoring="neg_mean_absolute_error")

    print_ensemble_results(voting_scores, voting_weights_scores, stacking_scores, stacking_all_scores)

    gb_reg = best_models['GradientBoostingRegressor']
    gb_reg.fit(X_train, y_train.values.ravel())
    joblib.dump(gb_reg, 'fitted_gb_reg.joblib')
