from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from .base_model import BaseXGModel


class RandomForestXG(ClassifierMixin, BaseXGModel):
    """
    Random Forest implementation for xG prediction.
    
    This class wraps sklearn's RandomForestClassifier and provides
    validation for parameter combinations specific to Random Forest.
    """
    
    def __init__(self, random_state: int = 42, param_grid: dict[str, list] | None = None):
        """
        Initialize the RandomForestXG model.
        
        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducibility.
        param_grid : dict[str, list] | None
            Hyperparameter grid for search procedures.
        """
        super().__init__(random_state=random_state, param_grid=param_grid)
        self.name = "RandomForest"
    
    def _create_model(self, **params) -> BaseEstimator:
        """
        Create and return a RandomForestClassifier estimator.
        
        Parameters
        ----------
        **params
            Hyperparameters forwarded to RandomForestClassifier constructor.
            
        Returns
        -------
        BaseEstimator
            Initialized RandomForestClassifier instance.
        """
        params['random_state'] = self.random_state
        # Set n_jobs to use all available cores for faster training
        if 'n_jobs' not in params:
            params['n_jobs'] = -1
            
        return RandomForestClassifier(**params)
    
    def is_valid_combination(self, params: dict[str, Any]) -> bool:
        """
        Validate Random Forest hyperparameter combinations.
        
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
        Random Forest validation rules:
        - min_samples_split must be >= 2
        - min_samples_leaf must be >= 1
        - max_features must be positive if specified as int
        - max_depth must be positive if not None
        """
        # Check min_samples_split
        if 'min_samples_split' in params:
            min_split = params['min_samples_split']
            if isinstance(min_split, int) and min_split < 2:
                return False
            elif isinstance(min_split, float) and (min_split <= 0 or min_split > 1):
                return False
        
        # Check min_samples_leaf
        if 'min_samples_leaf' in params:
            min_leaf = params['min_samples_leaf']
            if isinstance(min_leaf, int) and min_leaf < 1:
                return False
            elif isinstance(min_leaf, float) and (min_leaf <= 0 or min_leaf > 0.5):
                return False
        
        # Check max_features
        if 'max_features' in params:
            max_feat = params['max_features']
            if isinstance(max_feat, int) and max_feat < 1:
                return False
        
        # Check max_depth
        if 'max_depth' in params:
            max_depth = params['max_depth']
            if max_depth is not None and max_depth < 1:
                return False
        
        # Check n_estimators
        if 'n_estimators' in params:
            n_est = params['n_estimators']
            if n_est < 1:
                return False
        
        return True