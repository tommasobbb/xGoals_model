from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss


class BaseXGModel(ABC):
    """
    Abstract base class for expected-goals (xG) prediction models.

    This class defines a consistent interface for all xG models, including
    hyperparameter management, model fitting and prediction, performance
    evaluation, and (optionally) feature importance extraction.
    """

    def __init__(self, random_state: int = 42, param_grid: dict[str, list] | None = None):
        """
        Initializes the BaseXGModel class.

        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducibility.
        param_grid : dict[str, list] | None
            Hyperparameter grid for search procedures. If not provided,
            `get_param_grid` will raise.
        """
        self.random_state = random_state
        self.model: BaseEstimator | None = None
        self.is_fitted = False
        self._param_grid = param_grid
        self._estimator_type = "classifier"
    
    @abstractmethod
    def _create_model(self, **params) -> BaseEstimator:
        """
        Create and return the underlying estimator.

        Parameters
        ----------
        **params
            Hyperparameters forwarded to the estimator constructor.

        Returns
        -------
        BaseEstimator
            Initialized (unfitted) estimator instance.
        """

        pass

    @abstractmethod
    def is_valid_combination(self, params: dict[str, Any]) -> bool:
        """
        Validate a hyperparameter combination.

        Parameters
        ----------
        params : dict[str, Any]
            Candidate hyperparameters to validate.

        Returns
        -------
        bool
            True if the combination is valid; False otherwise.

        Notes
        -----
        Override in subclasses to enforce model-specific constraints
        (e.g., mutually exclusive parameters).
        """
        pass
    
    def get_param_grid(self) -> dict[str, list]:
        """
        Return the hyperparameter grid.

        Returns
        -------
        dict[str, list]
            Mapping of parameter names to lists of candidate values.

        Raises
        ------
        ValueError
            If no parameter grid was provided at initialization.
        """
        if self._param_grid is None:
            raise ValueError("Parameter grid not provided in config.yaml.")
        return self._param_grid
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> 'BaseXGModel':
        """
        Fit the model to training data.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix of shape (n_samples, n_features).
        y : pandas.Series
            Binary target vector (1 = goal, 0 = no goal) of shape (n_samples,).
        **fit_params
            Additional keyword arguments forwarded to the estimator's `fit`.

        Returns
        -------
        BaseXGModel
            The fitted instance (enables method chaining).

        Raises
        ------
        ValueError
            If the underlying model has not been initialized via `set_params`.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call set_params() first.")
        
        self.classes_ = unique_labels(y) 
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples,) with predicted labels {0, 1}.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        self._check_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples, 2) with probabilities for
            `[no_goal, goal]` in columns 0 and 1, respectively.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """

        self._check_fitted()
        proba = self.model.predict_proba(X)

        if hasattr(self.model, "classes_"):
            order = [int(np.where(self.model.classes_ == c)[0][0]) for c in self.classes_]
            proba = proba[:, order]

        return proba
    
    def predict_xg(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict xG values (probability of scoring).

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples,) with probabilities of the positive
            class (goal).
        """
        return self.predict_proba(X)[:, 1]  # Probability of positive class (goal)
    
    def set_params(self, **params) -> 'BaseXGModel':
        """
        Set hyperparameters and (re)initialize the underlying estimator.

        Parameters
        ----------
        **params
            Hyperparameters forwarded to `_create_model`.

        Returns
        -------
        BaseXGModel
            The instance with a new (unfitted) estimator configured.

        Notes
        -----
        This resets the fitted state. Call `fit` after updating parameters.
        """
        
        # Create new model with these parameters
        self.model = self._create_model(**params)
        self.is_fitted = False
        return self
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, float], np.ndarray]:
        """
        Compute evaluation metrics on a labeled dataset.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix of shape (n_samples, n_features).
        y : pandas.Series
            True binary labels of shape (n_samples,).

        Returns
        -------
        dict[str, float]
            Dictionary with the following keys:
            - 'accuracy'
            - 'precision'
            - 'recall'
            - 'f1'
            - 'roc_auc'

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Notes
        -----
        `precision`, `recall`, and `f1` handle zero divisions by returning 0.
        `roc_auc` is computed from predicted probabilities.
        """
        self._check_fitted()
        
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_xg(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'brier_score': brier_score_loss(y, y_proba)
        }
        
        return metrics, y_proba
    
    def get_feature_importance(self) -> np.ndarray | None:
        """
        Return feature importances, if supported by the estimator.

        Returns
        -------
        numpy.ndarray | None
            Importance scores per feature, or ``None`` if not available.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Notes
        -----
        If the estimator exposes `feature_importances_`, those values are
        returned. If it exposes `coef_` (e.g., linear models), the absolute
        values of the coefficients are returned.
        """

        self._check_fitted()
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients as importance
            return np.abs(self.model.coef_[0])
        else:
            return None
    
    def _check_fitted(self):
        """
        Validate that the model has been fitted.

        Raises
        ------
        ValueError
            If the estimator is not fitted yet.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model is not fitted yet. Call fit() first.")