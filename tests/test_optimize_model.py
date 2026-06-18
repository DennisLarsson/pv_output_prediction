import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from optimize_model import get_search_spaces, create_stacking_regressor, print_ensemble_results


class TestOptimizeModel(unittest.TestCase):

    def test_get_search_spaces(self):
        search_spaces = get_search_spaces()
        self.assertIsInstance(search_spaces, dict)
        self.assertIn('LinearRegression', search_spaces)
        self.assertIn('DecisionTreeRegressor', search_spaces)
        self.assertIn('RandomForestRegressor', search_spaces)

    def test_create_stacking_regressor(self):
        # Mock models
        mock_models = [
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=10))
        ]

        # Test with all_model=False
        stacking_regressor = create_stacking_regressor(mock_models, all_model=False, final_estimator='LinearRegression')
        self.assertIsNotNone(stacking_regressor)

        # Test with all_model=True
        stacking_regressor_all = create_stacking_regressor(mock_models, all_model=True, final_estimator='Ridge')
        self.assertIsNotNone(stacking_regressor_all)

    @patch('builtins.print')
    def test_print_ensemble_results(self, mock_print):
        # Mock scores
        voting_scores = np.array([-1.0, -2.0, -3.0])
        voting_weights_scores = np.array([-1.5, -2.5, -3.5])
        stacking_scores = np.array([-0.5, -1.5, -2.5])
        stacking_all_scores = np.array([-0.75, -1.75, -2.75])

        # Call the function
        print_ensemble_results(voting_scores, voting_weights_scores, stacking_scores, stacking_all_scores)

        # Check if print was called
        self.assertTrue(mock_print.called)


if __name__ == '__main__':
    unittest.main()