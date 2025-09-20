from typing import Any
import itertools
import numpy as np
import pandas as pd
from ..models.base_model import BaseXGModel
from .cv_strategy import CVStrategy


class GridSearchCV:
    """
    Custom grid search with cross-validation for xG models.
    
    Performs exhaustive search over parameter grid with consistent CV strategy.
    Tracks detailed results for analysis and comparison.
    """
    
    def __init__(
        self,
        model: BaseXGModel,
        cv_strategy: CVStrategy,
        scoring: str = 'brier_score',
        verbose: bool = True,
    ):
        self.model = model
        self.cv_strategy = cv_strategy
        self.scoring = scoring
        self.verbose = verbose

        # Results storage
        self.results_: list[dict[str, Any]] = []
        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = np.inf
        self.best_model_: BaseXGModel = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'GridSearchCV':
        """
        Perform grid search with cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : pd.Series
            Training target vector.

        Returns
        -------
        GridSearchCV
            Self, to allow method chaining.
        """
        param_grid = self.model.get_param_grid()
        param_combinations = self._generate_param_combinations(param_grid)

        if self.verbose:
            print(f"Starting grid search")
            print(f"Total parameter combinations: {len(param_combinations)}")
            print(f"CV folds: {self.cv_strategy.n_splits}")
            print(f"Total fits: {len(param_combinations) * self.cv_strategy.n_splits}")
            print("-" * 50)
        
        for i, params in enumerate(param_combinations):
            if self.verbose and (i + 1) % max(1, len(param_combinations) // 10) == 0:
                print(f"Progress: {i + 1}/{len(param_combinations)} combinations")
            
            # Evaluate this parameter combination
            cv_scores, cv_detailed = self._evaluate_params(X, y, params)
            
            # Calculate summary statistics
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # Store results
            result = {
                'params': params.copy(),
                'cv_scores': cv_scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'detailed_metrics': cv_detailed
            }
            self.results_.append(result)
            
            # Update best if this is better
            if mean_score < self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params.copy()
        
        # Train final model with best parameters
        self.best_model_ = self.model.__class__(random_state=self.model.random_state)
        self.best_model_.set_params(**self.best_params_)
        
        if self.verbose:
            print(f"\nBest {self.scoring}: {self.best_score_:.4f}")
            print(f"Best parameters: {self.best_params_}")
        
        return self
    
    def _generate_param_combinations(self, param_grid: dict[str, list]) -> list[dict[str, Any]]:
        """
        Generate all valid parameter combinations from the grid.

        Invalid combinations, as defined by the model's `is_valid_combination` method,
        are excluded from the returned list.

        Parameters
        ----------
        param_grid : dict of str to list
            Dictionary defining hyperparameter names and their possible values.

        Returns
        -------
        list of dict
            List of valid parameter combinations.
        """
        # Get all parameter names and their possible values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            if self.model.is_valid_combination(param_dict):
                combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_params(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        params: dict[str, Any]
    ) -> tuple[list[float], list[dict[str, float]]]:
        """
        Evaluate a specific set of parameters using cross-validation.

        A new model is trained and evaluated on each fold of the CV strategy.
        Evaluation metrics are collected per fold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        params : dict
            Hyperparameters to evaluate.

        Returns
        -------
        tuple of list of float, list of dict
            - List of primary scores (e.g., ROC AUC) for each fold.
            - List of detailed evaluation metrics for each fold.
        """

        cv_scores = []
        cv_detailed = []
        
        for fold_data in self.cv_strategy.cv_splits(X, y):
            X_train_fold, X_val_fold, y_train_fold, y_val_fold, scaler = fold_data
            
            # Create and train model for this fold
            fold_model = self.model.__class__(random_state=self.model.random_state)
            fold_model.set_params(**params)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Evaluate on validation fold
            metrics, y_proba = fold_model.evaluate(X_val_fold, y_val_fold)
            
            # Store primary score and all metrics
            primary_score = metrics.get(self.scoring, metrics['brier_score'])
            cv_scores.append(primary_score)
            cv_detailed.append(metrics)
        
        return cv_scores, cv_detailed
    
    def get_results_df(self) -> pd.DataFrame:
        """
        Get the results of the grid search as a DataFrame.

        The resulting DataFrame includes the mean and standard deviation of
        the primary score across folds, along with the associated hyperparameters.

        Returns
        -------
        pd.DataFrame
            Grid search results, sorted by descending mean score.
        """
        if not self.results_:
            return pd.DataFrame()
        
        # Flatten results for DataFrame
        rows = []
        for result in self.results_:
            row = {
                'mean_score': result['mean_score'],
                'std_score': result['std_score'],
            }
            # Add parameters as separate columns
            row.update(result['params'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by score (descending)
        df = df.sort_values('mean_score', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_best_model(self) -> BaseXGModel:
        """
        Retrieve the best model trained on the full training data.

        Raises
        ------
        ValueError
            If `fit` has not been called and no model has been trained.

        Returns
        -------
        BaseXGModel
            The best model found during grid search.
        """
        if self.best_model_ is None:
            raise ValueError("Grid search not fitted yet. Call fit() first.")
        return self.best_model_