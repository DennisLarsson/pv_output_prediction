from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, VotingRegressor, StackingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import clone
import numpy as np
import joblib
import argparse
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

def create_models(seed=None):
    print("Creating models...")
    scaled_models = {
        'LinearSVR': (Pipeline([('scaler', StandardScaler()), ('model', LinearSVR(random_state=seed))]),
                      get_search_spaces()['LinearSVR']),
        'SGDRegressor': (Pipeline([('scaler', StandardScaler()), ('model', SGDRegressor(random_state=seed))]),
                         get_search_spaces()['SGDRegressor']),
        'Ridge': (Pipeline([('scaler', StandardScaler()), ('model', Ridge(random_state=seed))]), get_search_spaces()['Ridge']),
        'Lasso': (Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=seed))]), get_search_spaces()['Lasso']),
        'ElasticNet': (Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(random_state=seed))]),
                       get_search_spaces()['ElasticNet']),
    }

    unscaled_models = {
        'LinearRegression': (LinearRegression(), get_search_spaces()['LinearRegression']),
        'DecisionTreeRegressor': (DecisionTreeRegressor(random_state=seed), get_search_spaces()['DecisionTreeRegressor']),
        'RandomForestRegressor': (RandomForestRegressor(random_state=seed), get_search_spaces()['RandomForestRegressor']),
        'ExtraTreesRegressor': (ExtraTreesRegressor(random_state=seed), get_search_spaces()['ExtraTreesRegressor']),
        'AdaBoostRegressor': (AdaBoostRegressor(random_state=seed), get_search_spaces()['AdaBoostRegressor']),
        'GradientBoostingRegressor': (GradientBoostingRegressor(random_state=seed), get_search_spaces()['GradientBoostingRegressor']),
    }

    models = {**scaled_models, **unscaled_models}
    return models

def run_bayes_search(model_set, n_iter=50, threads = -1, seed=None):
    print("Running Bayes Search...")
    results = {}
    for name, (model, search_space) in model_set.items():
        if not search_space:
            results[name] = {'model': model, 'best_params': {}, 'best_score': None}
            continue

        kf = KFold(n_splits=3, shuffle=True, random_state=seed)
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=kf,
            scoring="neg_mean_absolute_error",
            n_jobs=threads,
            random_state=seed,
        )

        opt.fit(X_train, y_train.values.ravel())

        results[name] = {
            'model': opt,
            'best_params': opt.best_params_,
            'best_score': opt.best_score_,
        }

    if verbose:
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

def get_weights(best_results):
    inv_scores = {name: 1 / (1 + score['best_score']) for name, score in best_results.items()}
    total = sum(inv_scores.values())
    weights = [inv_scores[name] / total for name in best_models]

    return weights

def create_voting_regressor(models, weights=None, threads=-1):
    print("Creating voting regressor...")
    voting_regressor = VotingRegressor(
        estimators=[(name, model) for name, model in models.items()],
        n_jobs=threads,
        weights=weights,
    )

    return voting_regressor

def create_stacking_regressor(best_models, all_model=True, final_estimator='Ridge',
                              seed=None, threads=-1):
    print("Creating stacking regressor...")
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    if all_model:
        stacking_regressor = StackingRegressor(
            estimators=[(name, model) for name, model in best_models.items()],
            final_estimator=clone(best_models[final_estimator]),
            cv=kf,
            n_jobs=threads
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
            cv=kf,
            n_jobs=threads
        )

    return stacking_regressor

def run_cross_val_score(estimator, X, y, seed=None):
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    scores = cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        cv=kf,
        scoring="neg_mean_absolute_error"
    )

    return estimator, scores

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

    parser = argparse.ArgumentParser()
    parser.add_argument('pvgis_filename')
    parser.add_argument('-n','--bayes_n_iter', default=50)
    parser.add_argument('-o', '--output', default='fitted_best_model.joblib')
    parser.add_argument('-s', '--seed', default=None)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('-T', '--threads', default=-1)

    args = parser.parse_args()
    pvgis_filename = args.pvgis_filename
    bayes_n_iter = int(args.bayes_n_iter)
    output_file = args.output
    seed = int(args.seed)
    verbose = args.verbose
    threads = int(args.threads)

    X_train, X_test, y_train, y_test = process_pvgis(pvgis_filename)

    models = create_models()
    best_results = run_bayes_search(models, n_iter = bayes_n_iter, threads=threads, seed=seed)
    best_models = get_best_models(best_results)

    weights = get_weights(best_results)

    voting_regressor, voting_scores = run_cross_val_score(
        create_voting_regressor(best_models, threads=threads),
        X_train,
        y_train.values.ravel(),
        seed=seed
    )

    voting_regressor_weights, voting_weights_scores = run_cross_val_score(
        create_voting_regressor(best_models, weights, threads=threads),
        X_train,
        y_train.values.ravel(),
        seed=seed
    )

    stacking_regressor, stacking_scores = run_cross_val_score(
        create_stacking_regressor(models, all_model=False, final_estimator='RandomForestRegressor', seed=seed, threads=threads),
        X_train,
        y_train.values.ravel(),
        seed=seed
    )

    stacking_regressor_all, stacking_all_scores = run_cross_val_score(
        create_stacking_regressor(models, all_model=True, final_estimator='Ridge', seed=seed, threads=threads),
        X_train,
        y_train.values.ravel(),
        seed=seed
    )

    if verbose:
        print_ensemble_results(voting_scores, voting_weights_scores, stacking_scores, stacking_all_scores)

    ensemble_results = {
        'VotingRegressor': {
            'model': voting_regressor,
            'best_score': np.mean(voting_weights_scores)
        },
        'VotingRegressorWeights': {
            'model': voting_regressor_weights,
            'best_score': np.mean(voting_weights_scores)
        },
        'StackingRegressor': {
            'model': stacking_regressor,
            'best_score': np.mean(stacking_scores)
        },
        'StackingRegressorAll': {
            'model': stacking_regressor_all,
            'best_score': np.mean(stacking_all_scores)
        }
    }

    best_results = {**best_results, **ensemble_results}

    best_score = -np.inf
    best_model_name = ""
    for name, model in best_results.items():
        if model['best_score'] > best_score:
            best_score = model['best_score']
            final_model = model['model']
            best_model_name = name

    print("Best Model:", best_model_name)

    final_model.fit(X_train, y_train.values.ravel())
    joblib.dump(final_model, output_file)
