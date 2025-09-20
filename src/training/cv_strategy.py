from typing import Iterator
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


class CVStrategy:
    """
    Centralized cross-validation strategy for xG models.
    
    Ensures all models use the same CV splits for fair comparison.
    Handles data preprocessing (scaling) within each fold to prevent leakage.
    """
    
    def __init__(
        self, 
        n_splits: int = 5,
        random_state: int = 42,
        test_size: float = 0.2,
        scale_features: bool = True
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.test_size = test_size
        self.scale_features = scale_features
        
        # Initialize CV splitter
        self.cv_splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True, 
            random_state=random_state
        )
        
    def train_test_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into training and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        tuple of pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
            X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
    
    def cv_splits(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]]:
        """
        Generate stratified cross-validation splits with optional feature scaling.

        For each fold, training and validation sets are yielded along with a
        fitted `StandardScaler`, if scaling is enabled. The scaler is fitted
        only on the training fold to avoid data leakage.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (training set only).
        y : pd.Series
            Target vector (training set only).

        Yields
        ------
        tuple of pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler or None
            X_train_fold, X_val_fold, y_train_fold, y_val_fold, scaler
        """
        # Get feature columns (exclude non-feature columns)
        feature_cols = self._get_feature_columns(X)
        
        for train_idx, val_idx in self.cv_splitter.split(X, y):
            # Split data
            X_train_fold = X.iloc[train_idx].copy()
            X_val_fold = X.iloc[val_idx].copy()
            y_train_fold = y.iloc[train_idx].copy()
            y_val_fold = y.iloc[val_idx].copy()
            
            # Scale features if requested
            scaler = None
            if self.scale_features:
                scaler = StandardScaler()
                
                # Fit scaler on training fold only
                X_train_fold[feature_cols] = scaler.fit_transform(
                    X_train_fold[feature_cols]
                )
                
                # Transform validation fold
                X_val_fold[feature_cols] = scaler.transform(
                    X_val_fold[feature_cols]
                )
            
            yield X_train_fold, X_val_fold, y_train_fold, y_val_fold, scaler
    
    def prepare_final_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list]:
        """
        Scale the final training and test sets using a scaler fit on the full training set.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        X_test : pd.DataFrame
            Test feature matrix.
        y_train : pd.Series
            Training target vector. (Used only for reference.)

        Returns
        -------
        tuple of pd.DataFrame, pd.DataFrame, StandardScaler or None
            Scaled X_train, scaled X_test, and the fitted scaler.
        """
        X_train_final = X_train.copy()
        X_test_final = X_test.copy()
        
        scaler = None
        if self.scale_features:
            feature_cols = self._get_feature_columns(X_train)
            
            scaler = StandardScaler()
            X_train_final[feature_cols] = scaler.fit_transform(X_train[feature_cols])
            X_test_final[feature_cols] = scaler.transform(X_test[feature_cols])
        
        return X_train_final, X_test_final, scaler, feature_cols
    
    def _get_feature_columns(self, X: pd.DataFrame) -> list:
        """
        Identify feature columns eligible for scaling.

        Excludes identifier and non-numeric columns (e.g., 'match_id').

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        list
            List of column names to be scaled.
        """
        # Exclude known non-feature columns
        exclude_cols = ['match_id']
        
        # Select numeric columns that aren't excluded
        feature_cols = [
            col for col in X.columns 
            if col not in exclude_cols and X[col].dtype in ['float64', 'int64', 'uint8']
        ]
        
        return feature_cols