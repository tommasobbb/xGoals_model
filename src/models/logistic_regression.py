from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from .base_model import BaseXGModel


class LogisticRegressionXG(BaseXGModel):
    """
    Logistic Regression model for xG prediction.
    """
    
    def __init__(self, random_state: int = 42, param_grid: dict[str, list] | None = None):
        super().__init__(random_state, param_grid)
    
    def _create_model(self, **params) -> BaseEstimator:
        """
        Create LogisticRegression model with given parameters.
        """        
        return LogisticRegression(
            random_state=self.random_state,
            **params
        )
    
    def is_valid_combination(self, params: dict[str, Any]) -> bool:
        """ Validates LogisticRegression hyperparameter combinations
        based on the user's specified param_grid.
        """
        pen = params.get("penalty")
        sol = params.get("solver") # Check for L1 penalty and solver compatibility
        
        if pen == "l1":
            if sol != "liblinear":
                return False
        
        return True