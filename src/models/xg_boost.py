from typing import Any
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from .base_model import BaseXGModel


class XGBoostXG(ClassifierMixin, BaseXGModel):
    """
    XGBoost implementation for xG prediction.
    
    This class wraps XGBoost's XGBClassifier and provides
    validation for parameter combinations specific to XGBoost.
    """
    _estimator_type = "classifier"
    
    def __init__(self, random_state: int = 42, param_grid: dict[str, list] | None = None):
        """
        Initialize the XGBoostXG model.
        
        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducibility.
        param_grid : dict[str, list] | None
            Hyperparameter grid for search procedures.
            
        Raises
        ------
        ImportError
            If XGBoost is not installed.
        """
        super().__init__(random_state=random_state, param_grid=param_grid)
        self.name = "XGBoost"
    
    def _create_model(self, **params) -> BaseEstimator:
        """
        Create and return an XGBClassifier estimator.
        
        Parameters
        ----------
        **params
            Hyperparameters forwarded to XGBClassifier constructor.
            
        Returns
        -------
        BaseEstimator
            Initialized XGBClassifier instance.
        """
        # Set default parameters for xG prediction
        default_params = {
            'random_state': self.random_state,
            'objective': 'binary:logistic',  # For binary classification
            'eval_metric': 'auc',        # Evaluation metric
            'n_jobs': -1,                    # Use all available cores
            'verbosity': 0                   # Suppress XGBoost output
        }
        
        # Update with user parameters (user params override defaults)
        default_params.update(params)
        
        return XGBClassifier(**default_params)
    
    def is_valid_combination(self, params: dict[str, Any]) -> bool:
        """
        Validate XGBoost hyperparameter combinations.
        
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
        XGBoost validation rules:
        - learning_rate (eta) must be > 0
        - max_depth must be >= 0
        - min_child_weight must be >= 0
        - subsample must be in (0, 1]
        - colsample_bytree must be in (0, 1]
        - n_estimators must be > 0
        - reg_alpha and reg_lambda must be >= 0
        """
        # Check learning_rate
        if 'learning_rate' in params:
            lr = params['learning_rate']
            if lr <= 0:
                return False
        
        # Check max_depth
        if 'max_depth' in params:
            max_depth = params['max_depth']
            if max_depth < 0:
                return False
        
        # Check min_child_weight
        if 'min_child_weight' in params:
            mcw = params['min_child_weight']
            if mcw < 0:
                return False
        
        # Check subsample
        if 'subsample' in params:
            subsample = params['subsample']
            if subsample <= 0 or subsample > 1:
                return False
        
        # Check colsample_bytree
        if 'colsample_bytree' in params:
            colsample = params['colsample_bytree']
            if colsample <= 0 or colsample > 1:
                return False
        
        # Check n_estimators
        if 'n_estimators' in params:
            n_est = params['n_estimators']
            if n_est <= 0:
                return False
        
        # Check regularization parameters
        for reg_param in ['reg_alpha', 'reg_lambda']:
            if reg_param in params:
                reg_val = params[reg_param]
                if reg_val < 0:
                    return False
        
        # Check gamma
        if 'gamma' in params:
            gamma = params['gamma']
            if gamma < 0:
                return False
        
        return True