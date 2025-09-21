from typing import Any
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from .base_model import BaseXGModel


class MLPClassifierXG(ClassifierMixin, BaseXGModel):
    """
    Multilayer Perceptron (feedforward neural network) for expected goals (xG) prediction.

    This class wraps scikit-learn's ``MLPClassifier`` for use in the xG modeling
    framework. It inherits from ``BaseXGModel`` and provides additional validation
    for hyperparameter combinations.
    """

    def __init__(self, random_state: int = 42,
                 param_grid: dict[str, list] | None = None):
        """
        Initialize the MLPClassifier-based xG model.

        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducibility.
        param_grid : dict of str -> list, optional
            Hyperparameter search grid for model tuning.
        """
        super().__init__(random_state, param_grid)

    def _create_model(self, **params) -> BaseEstimator:
        """
        Create an instance of ``MLPClassifier`` with the given parameters.

        Parameters
        ----------
        **params : dict
            Keyword arguments corresponding to valid ``MLPClassifier``
            hyperparameters.

        Returns
        -------
        model : MLPClassifier
            Instantiated scikit-learn ``MLPClassifier`` with the provided
            parameters and fixed ``random_state``.
        """
        return MLPClassifier(
            random_state=self.random_state,
            **params
        )

    def is_valid_combination(self, params: dict[str, Any]) -> bool:
        """
        Validate hyperparameter combinations for ``MLPClassifier``.

        Ensures that the provided parameters are consistent with scikit-learn's
        documented behavior. In particular, checks compatibility between solver,
        early stopping, learning rate schedules, and momentum.

        Parameters
        ----------
        params : dict of str -> Any
            Dictionary of hyperparameters for ``MLPClassifier``.

        Returns
        -------
        is_valid : bool
            ``True`` if the parameter combination is valid, ``False`` otherwise.
        """
        solver = params.get("solver", "adam")
        early_stopping = params.get("early_stopping", False)
        learning_rate = params.get("learning_rate", "constant")
        momentum = params.get("momentum", None)

        if solver not in {"lbfgs", "sgd", "adam"}:
            return False

        if early_stopping and solver == "lbfgs":
            return False

        if solver != "sgd" and "learning_rate" in params and learning_rate != "constant":
            return False

        if momentum is not None and solver != "sgd":
            return False

        activation = params.get("activation", "relu")
        if activation not in {"identity", "logistic", "tanh", "relu"}:
            return False

        alpha = params.get("alpha", 0.0001)
        try:
            if float(alpha) < 0:
                return False
        except Exception:
            return False

        max_iter = params.get("max_iter", 200)
        try:
            if int(max_iter) <= 0:
                return False
        except Exception:
            return False

        return True
