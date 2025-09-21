import argparse
import json
import pickle
import time
from typing import Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from ..config import FEATURES_OUT_PATH, RESULTS_PATH, TRAINED_MODELS_PATH
from ..models.logistic_regression import LogisticRegressionXG
from ..models.xg_boost import XGBoostXG
from ..models.random_forest import RandomForestXG
from ..models.mlp_classifier import MLPClassifierXG

from ..training.cv_strategy import CVStrategy
from ..training.grid_search import GridSearchCV


# Model registry
MODEL_REGISTRY = {
    'logistic_regression': LogisticRegressionXG,
    'xgboost': XGBoostXG,
    'random_forest': RandomForestXG,
    'mlp_classifier': MLPClassifierXG,
}

class SingleModelTrainer:
    """
    Trains a single xG prediction model with grid search and cross-validation.
    
    Handles data loading, train/test split, grid search with CV,
    final model training, and results saving.
    """
    
    def __init__(self, model_name: str, config: dict[str, Any]):
        """
        Initializes the SingleModelTrainer.

        Parameters
        ----------
        model_name : str
            The name of the model to be trained. It must be a key in the
            `MODEL_REGISTRY`.
        config : dict
            A dictionary containing the full training configuration,
            loaded from a YAML file.
        """
        if model_name not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        
        self.model_name = model_name
        self.config = config
        
        # Initialize CV strategy with settings from the config
        self.cv_strategy = CVStrategy(
            n_splits=self.config['training']['cv_folds'],
            random_state=self.config['training']['random_state'],
            test_size=self.config['training']['test_size'],
            scale_features=self.config['training']['scale_features']
        )
        
        # Create output directories using paths from the config
        RESULTS_PATH.mkdir(exist_ok=True)
        TRAINED_MODELS_PATH.mkdir(exist_ok=True)
        
        print(f"🎯 Training: {model_name}")
        print(f"📁 Results will be saved to: {RESULTS_PATH}")
        print(f"💾 Models will be saved to: {TRAINED_MODELS_PATH}")

    
    def load_data(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads and prepares the processed feature data for training.

        Reads the final Parquet file containing the engineered features,
        separates the features from the target variable, and ensures the
        target is in a binary format.

        Parameters
        ----------
        self : object
            The instance of the `SingleModelTrainer` class.

        Returns
        -------
        tuple
            A tuple containing:
            - X : pandas.DataFrame
                The feature matrix.
            - y : pandas.Series
                The binary target vector.
            - sb_xg : pandas.Series
                The StatsBomb xG values (not used for training).
        """
        print("\n📊 Loading processed features...")
        df = pd.read_parquet(FEATURES_OUT_PATH)

        # id must exist and be unique
        assert 'id' in df.columns, "Missing 'id' column"
        assert df['id'].is_unique, "'id' must be unique per row"
        df = df.set_index('id')

        # Pull out target and StatsBomb xG
        y = df['is_goal'].astype(int)
        sb_xg = df['shot_statsbomb_xg'].astype(float) if 'shot_statsbomb_xg' in df.columns else None

        # Remove target & statsbomb_xg from features
        drop_cols = ['is_goal', 'match_id']
        if 'shot_statsbomb_xg' in df.columns:
            drop_cols.append('shot_statsbomb_xg')

        X = df.drop(columns=drop_cols, errors='ignore')

        print(f"   • Total shots: {len(df)}")
        print(f"   • Features: {len(X.columns)}")
        print(f"   • Goal rate: {y.mean():.3f} ({y.sum()}/{len(y)} goals)")
        if sb_xg is not None:
            print(f"   • StatsBomb xG present. Will compare on test set.")

        return X, y, sb_xg


    def _evaluate_final_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> tuple[dict[str, float], np.ndarray]:
        """
        Compute evaluation metrics for a fitted classifier.

        Parameters
        ----------
        model : Any
            A fitted sklearn-style model supporting predict / predict_proba.
            Works with both base models and CalibratedClassifierCV.
        X : pd.DataFrame
            Feature matrix of shape (n_samples, n_features).
        y : pd.Series
            True binary labels of shape (n_samples,).

        Returns
        -------
        metrics : dict[str, float]
            Dictionary with model metrics.
        y_proba : np.ndarray
            Predicted probabilities for the positive class, shape (n_samples,).
        """
        if not hasattr(model, "predict"):
            raise ValueError("Model is not fitted or does not implement predict().")

        # Predictions
        y_pred = model.predict(X)

        # Probabilities for the positive class (assume binary classification)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            raise ValueError("Model does not support predict_proba().")

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
            "brier_score": brier_score_loss(y, y_proba),
        }

        return metrics, y_proba

    def train(self) -> None:
        """
        Trains the specified model using cross-validation and grid search.

        This function orchestrates the entire model training pipeline. It loads
        the data, performs a train-test split, uses a grid search with cross-validation
        to find the best hyperparameters, trains the final model on the full training
        set, evaluates its performance on the test set, and saves the results.
        """
        print("\n" + "=" * 60)
        print(f"🚀 TRAINING {self.model_name.upper()}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load data
        X, y, sb_xg = self.load_data()

        # Get model configurations
        model_config = self.config['models'][self.model_name]
        param_grid = model_config['param_grid']
        is_calibration_needed = model_config['is_calibration_needed']

        # Train/test split
        X_train, X_test, y_train, y_test = self.cv_strategy.train_test_split(X, y)

        # Optional calibration split — split *training only*
        if is_calibration_needed:
            X_subtr, X_cal, y_subtr, y_cal = train_test_split(
                X_train, y_train,
                test_size=0.20,
                stratify=y_train,
                random_state=42
            )
        else:
            X_subtr, y_subtr = X_train, y_train
            X_cal = X_cal_final = y_cal = None

        # Align StatsBomb xG with the same split
        sb_xg_train = sb_xg.loc[X_train.index] if sb_xg is not None else None
        sb_xg_test  = sb_xg.loc[X_test.index]  if sb_xg is not None else None
            
        print(f"\n📈 Data split:")
        print(f"   • Training: {len(X_train)} samples ({y_train.mean():.3f} goal rate)")
        print(f"   • Test: {len(X_test)} samples ({y_test.mean():.3f} goal rate)")
        
        # Initialize model
        model_class = MODEL_REGISTRY[self.model_name]
        model = model_class(random_state=self.config['training']['random_state'], param_grid=param_grid)
        
        # Grid search with CV
        print(f"\n🔍 Starting grid search with {self.config['training']['cv_folds']}-fold CV...")
        grid_search = GridSearchCV(
            model=model,
            cv_strategy=self.cv_strategy,
            scoring=self.config['training']['scoring_metric'],
            verbose=self.config['training']['verbose'],
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model and prepare final data
        best_model = grid_search.get_best_model()
            
        # Fit scaler on *subtrain* and transform subtrain/test (+ calibration if present)
        X_subtr_final, X_test_final, scaler, feature_cols = self.cv_strategy.prepare_final_data(
            X_subtr, X_test, y_subtr
        )
        if is_calibration_needed:
            X_cal_final = X_cal.copy()
            X_cal_final[feature_cols] = scaler.transform(X_cal[feature_cols])
        
        # Train final model on full training set
        print(f"\n🏆 Training final model with best parameters...")
        best_model.fit(X_subtr_final, y_subtr)
            
        if is_calibration_needed:
            print(f"\n🏆 Calibrating final model...")

            # Wrap base model in CalibratedClassifierCV (prefit)
            cal = CalibratedClassifierCV(best_model, method="isotonic",  cv="prefit")

            # Fit calibrator on calibration split
            cal.fit(X_cal_final, y_cal)

            calibrated_model = cal
        else:
            calibrated_model = best_model

        # Evaluate on test set
        print(f"📊 Evaluating on test set...")
        test_metrics, y_proba = self._evaluate_final_model(calibrated_model, X_test_final, y_test)

        statsbomb_xg_brier_score = brier_score_loss(y_test.values, sb_xg_test.values)
        print(f"StatsBomb xG Model Brier Score: {statsbomb_xg_brier_score:.4f}")

        # Get feature importance if available
        feature_importance = best_model.get_feature_importance()
        feature_names = [col for col in X_train.columns if col != 'match_id']
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Compile results
        results = {
            'model_name': self.model_name,
            'training_time_seconds': training_time,
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_info': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': len(feature_names),
                'goal_rate_train': float(y_train.mean()),
                'goal_rate_test': float(y_test.mean())
            },
            'cv_results': {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'scoring_metric': self.config['training']['scoring_metric'],
                'n_folds': self.config['training']['cv_folds'],
                'all_results': grid_search.get_results_df().to_dict('records')
            },
            'test_metrics': test_metrics,
            'feature_importance': {
                'values': feature_importance.tolist() if feature_importance is not None else None,
                'feature_names': feature_names if feature_importance is not None else None
            }
        }
        
        # Save results and model
        self._save_results(results, calibrated_model, scaler, feature_names)
        
        # Print summary
        self._print_results_summary(results)

        # Build comparison DataFrame
        comparison_df = pd.DataFrame({
            "statsbomb_xg": sb_xg_test.values,
            "our_model_xg": y_proba,
            "y_true": y_test.values
        }, index=X_test.index)

        # Save to CSV
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        comp_file = RESULTS_PATH / f"{self.model_name}_xg_comparison_{timestamp}.csv"
        comparison_df.to_csv(comp_file, index=False)

        print(f"📎 xG comparison CSV saved: {comp_file.name}")

        # Plot & save calibration curve
        calib_image = self._plot_calibration_curve(
            y_true=y_test.values,
            y_proba=y_proba,
            model_name=self.model_name,
            results_path=RESULTS_PATH,
            timestamp=timestamp,
            n_bins=self.config['training'].get('calibration_bins', 10),
            sb_xg=sb_xg_test.values if sb_xg_test is not None else None
        )
        print(f"🖼️  Calibration plot saved: {calib_image}")
                
    def _save_results(self, results: dict[str, Any], model, scaler, feature_names: list):
        """Saves the training results and the trained model.

        This function serializes and saves the complete training results to a
        JSON file, the trained model and its associated objects (scaler,
        feature names) to a pickle file, and a summary of key metrics to
        a CSV file.

        Parameters
        ----------
        self : object
            The instance of the `SingleModelTrainer` class.
        results : dict
            A dictionary containing all the training results.
        model : object
            The final trained model object.
        scaler : object
            The data scaler fitted to the training data.
        feature_names : list
            A list of the names of the features used to train the model.

        Notes
        -----
        The function generates a unique timestamp for each set of files to prevent
        overwriting previous runs.
        """
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        results_file = RESULTS_PATH/ f'{self.model_name}_results_{timestamp}.json'
        model_file = TRAINED_MODELS_PATH / f'{self.model_name}_model_{timestamp}.pkl'
                
        # Save detailed results (JSON)
        with results_file.open('w') as f:
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=2, default=str)
        
        # Save trained model (pickle)
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'model_name': self.model_name,
            'timestamp': timestamp,
            'best_params': results['cv_results']['best_params'],
            'test_metrics': results['test_metrics']
        }
        
        with model_file.open('wb') as f:
            pickle.dump(model_data, f)
        
        # Save quick summary (CSV)
        summary_data = {
            'model': [self.model_name],
            'timestamp': [timestamp],
            'cv_score': [results['cv_results']['best_cv_score']],
            'test_roc_auc': [results['test_metrics']['roc_auc']],
            'test_accuracy': [results['test_metrics']['accuracy']],
            'test_precision': [results['test_metrics']['precision']],
            'test_recall': [results['test_metrics']['recall']],
            'test_f1': [results['test_metrics']['f1']],
            'training_time_min': [results['training_time_seconds'] / 60]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = RESULTS_PATH / f'{self.model_name}_summary_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n💾 Files saved:")
        print(f"   • Results: {results_file.name}")
        print(f"   • Model: {model_file.name}")
        print(f"   • Summary: {summary_file.name}")
    
    def _prepare_for_json(self, obj):
        """
        Recursively prepares an object for JSON serialization.

        This helper method converts NumPy arrays and numeric types into standard
        Python lists and numbers to ensure they can be properly serialized into JSON format.

        Parameters
        ----------
        self : object
            The instance of the `SingleModelTrainer` class.
        obj : object
            The object to be prepared for JSON serialization.

        Returns
        -------
        object
            A JSON-serializable version of the input object.
        """
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _print_results_summary(self, results: dict[str, Any]):
        """
        Prints a summary of the training results to the console.

        This function formats and displays key performance metrics, the best
        hyperparameters found, and the top 10 most important features.

        Parameters
        ----------
        self : object
            The instance of the `SingleModelTrainer` class.
        results : dict
            A dictionary containing the training results.
        """
        print("\n" + "=" * 60)
        print("🎯 TRAINING COMPLETED")
        print("=" * 60)
        
        training_time = results['training_time_seconds']
        cv_score = results['cv_results']['best_cv_score']
        test_metrics = results['test_metrics']
        
        print(f"⏱️  Training time: {training_time/60:.1f} minutes")
        print(f"🏆 Best CV {self.config['training']['scoring_metric']}: {cv_score:.4f}")
        print("\n📊 Test Set Performance:")
        print(f"   • ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        print(f"   • Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"   • Precision: {test_metrics['precision']:.4f}")
        print(f"   • Recall:    {test_metrics['recall']:.4f}")
        print(f"   • F1-Score:  {test_metrics['f1']:.4f}")
        print(f"   • Brier-Score:  {test_metrics['brier_score']:.4f}")
        
        print(f"\n🔧 Best Parameters:")
        for param, value in results['cv_results']['best_params'].items():
            print(f"   • {param}: {value}")
        
        # Feature importance top 10
        if results['feature_importance']['values']:
            print(f"\n🏅 Top 10 Most Important Features:")
            importances = results['feature_importance']['values']
            names = results['feature_importance']['feature_names']
            
            # Sort by importance
            feature_imp = list(zip(names, importances))
            feature_imp.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for i, (name, imp) in enumerate(feature_imp[:10], 1):
                print(f"   {i:2d}. {name:25s} {imp:8.4f}")

    def _plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        results_path,
        *,
        timestamp: str | None = None,
        n_bins: int = 10,
        sb_xg: np.ndarray | None = None,
    ) -> str:
        """
        Plots and saves the calibration curve for a set of predicted probabilities.

        Parameters
        ----------
        y_true : array-like, shape (n_samples,)
            Ground-truth binary labels (0/1).
        y_proba : array-like, shape (n_samples,)
            Predicted probabilities for the positive class.
        model_name : str
            Name of the model (used in title/filename).
        results_path : pathlib.Path
            Directory where the image will be saved.
        timestamp : str, optional
            Timestamp string to include in filename. If None, uses current time.
        n_bins : int, default=10
            Number of bins to use for calibration curve.
        sb_xg : array-like, optional
            Optional comparator probabilities (e.g., StatsBomb xG) to plot as well.

        Returns
        -------
        str
            The saved image filename (not the full path).
        """
        # Safety: clip probs to [0,1]
        y_proba = np.clip(np.asarray(y_proba, dtype=float), 0.0, 1.0)
        y_true = np.asarray(y_true, dtype=int)

        if timestamp is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Perfectly calibrated')

        # Our model
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'{model_name}')

        # Optional comparator (e.g., StatsBomb xG)
        if sb_xg is not None:
            sb_xg = np.clip(np.asarray(sb_xg, dtype=float), 0.0, 1.0)
            sb_true, sb_pred = calibration_curve(y_true, sb_xg, n_bins=n_bins, strategy='uniform')
            ax.plot(sb_pred, sb_true, marker='s', linewidth=2, label='StatsBomb xG')

        ax.set_title(f'Calibration Curve — {model_name}')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.grid(alpha=0.25)
        ax.legend()

        # Twin axis for probability histograms
        ax_hist = ax.twinx()
        ax_hist.hist(
            y_proba, bins=20, range=(0, 1), alpha=0.25, density=True, label=f'{model_name} dist'
        )
        if sb_xg is not None:
            ax_hist.hist(
                sb_xg, bins=20, range=(0, 1), alpha=0.25, density=True, label='StatsBomb dist'
            )

        ax_hist.set_yticks([])
        ax_hist.set_ylabel('')
        ax_hist.legend(loc="upper center", frameon=False)

        fname = f'{model_name}_calibration_{timestamp}.png'
        fpath = results_path / fname
        fig.tight_layout()
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        
        return fname


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the training script.

    This function configures and parses the command-line arguments required to
    run the model training pipeline. It defines arguments for selecting the
    model and specifying the configuration file.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments as attributes.
        The attributes are:
        - `model` (str): The name of the model to be trained.
        - `config` (str): The path to the YAML configuration file.
    """
    parser = argparse.ArgumentParser(
        description="Train a single xG prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
    Available models:
    {chr(10).join(f'  • {name}' for name in MODEL_REGISTRY.keys())}

    Example usage:
    python -m src.scripts.train_models --model logistic_regression
    python -m src.scripts.train_models --model logistic_regression --config config.yaml
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help='Model to train'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to the configuration file (default: config.yaml)'
    )
    
    return parser.parse_args()


def main():
    """
    Executes the main model training pipeline.

    This function serves as the entry point for the training script.
    It parses command-line arguments, loads the specified configuration file,
    initializes the model trainer, and runs the training process.
    """
    args = parse_arguments()
    
    # Load the entire configuration from the YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize trainer and run, passing the loaded config dictionary
    trainer = SingleModelTrainer(args.model, config)
    trainer.train()

if __name__ == "__main__":
    main()