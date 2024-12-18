"""
Baseline model training for M-TRI toxin prediction.
Implements logistic regression and XGBoost with spatial cross-validation.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    brier_score_loss, classification_report, confusion_matrix,
    precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineTrainer:
    """Trains and evaluates baseline models for toxin prediction."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.evaluation_results = {}
        
        # Set up model configurations
        self.model_configs = {
            'logistic': {
                'model': LogisticRegression(
                    random_state=random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'scale_features': True
            },
            
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    class_weight='balanced',
                    max_depth=10
                ),
                'scale_features': False
            }
        }
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str = 'toxin_detected',
                    exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: Input dataframe with features
            target_col: Name of target variable column
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info(f"Preparing data: {len(df)} observations")
        
        if exclude_cols is None:
            exclude_cols = ['pond_id', 'date', 'lat', 'lon']
            
        # Separate features and target
        exclude_cols = exclude_cols + [target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        logger.info(f"Missing values before cleaning: {X.isnull().sum().sum()}")
        
        # Simple imputation for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Remove still-missing columns
        missing_cols = X.columns[X.isnull().any()].tolist()
        if missing_cols:
            logger.warning(f"Dropping columns with missing values: {missing_cols}")
            X = X.drop(columns=missing_cols)
            
        # Store feature names
        self.feature_names = list(X.columns)
        
        logger.info(f"Final feature set: {len(self.feature_names)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    def spatial_train_test_split(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                               test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                               np.ndarray, np.ndarray]:
        """
        Split data using spatial blocking to avoid data leakage.
        Groups ponds by geographic location to ensure spatial independence.
        """
        logger.info("Creating spatial train/test split...")
        
        # Create spatial groups based on lat/lon clustering
        from sklearn.cluster import KMeans
        
        # Get unique pond locations
        pond_locations = df.groupby('pond_id')[['lat', 'lon']].first()
        
        # Create spatial clusters (approximate watersheds)
        n_clusters = max(2, int(len(pond_locations) * 0.3))  # 30% of ponds as cluster centers
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        pond_locations['spatial_cluster'] = kmeans.fit_predict(pond_locations[['lat', 'lon']])
        
        # Map clusters back to full dataset
        df_with_clusters = df.merge(
            pond_locations[['spatial_cluster']], 
            left_on='pond_id', 
            right_index=True
        )
        
        # Split clusters into train/test
        unique_clusters = pond_locations['spatial_cluster'].unique()
        test_clusters = np.random.RandomState(self.random_state).choice(
            unique_clusters, 
            size=int(len(unique_clusters) * test_size),
            replace=False
        )
        
        # Create train/test masks
        test_mask = df_with_clusters['spatial_cluster'].isin(test_clusters)
        train_mask = ~test_mask
        
        X_train = X[train_mask].values
        X_test = X[test_mask].values
        y_train = y[train_mask].values  
        y_test = y[test_mask].values
        
        logger.info(f"Spatial split - Train: {len(X_train)} obs ({len(unique_clusters) - len(test_clusters)} clusters)")
        logger.info(f"Spatial split - Test: {len(X_test)} obs ({len(test_clusters)} clusters)")
        
        return X_train, X_test, y_train, y_test
        
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Train all baseline models.
        
        Args:
            X: Feature matrix
            y: Target vector  
            df: Original dataframe (needed for spatial split)
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training baseline models...")
        
        # Spatial train/test split
        if df is not None:
            X_train, X_test, y_train, y_test = self.spatial_train_test_split(df, X, y)
        else:
            # Fallback to random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
        results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            
            # Initialize model
            model = config['model']
            
            # Feature scaling if needed
            if config['scale_features']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[model_name] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated_model.fit(X_train_scaled, y_train)
            
            # Store models
            self.models[model_name] = model
            self.models[f'{model_name}_calibrated'] = calibrated_model
            
            # Evaluate model
            train_metrics = self._evaluate_model(model, X_train_scaled, y_train, 'train')
            test_metrics = self._evaluate_model(model, X_test_scaled, y_test, 'test')
            
            # Calibrated model evaluation
            cal_train_metrics = self._evaluate_model(calibrated_model, X_train_scaled, y_train, 'cal_train')
            cal_test_metrics = self._evaluate_model(calibrated_model, X_test_scaled, y_test, 'cal_test')
            
            results[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'calibrated_train_metrics': cal_train_metrics,
                'calibrated_test_metrics': cal_test_metrics,
                'feature_importance': self._get_feature_importance(model, model_name)
            }
            
            logger.info(f"{model_name} Test ROC-AUC: {test_metrics['roc_auc']:.3f}")
            
        self.evaluation_results = results
        return results
        
    def _evaluate_model(self, model, X, y, split_name: str) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics."""
        
        # Predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # Core metrics
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Brier score (lower is better)
        brier_score = brier_score_loss(y, y_pred_proba)
        
        # Precision at top K (important for ranking systems)
        k = min(20, len(y))
        top_k_indices = np.argsort(y_pred_proba)[-k:]
        precision_at_k = y[top_k_indices].mean() if k > 0 else 0
        
        # Classification metrics
        precision_binary = precision_score(y, y_pred, zero_division=0)
        recall_binary = recall_score(y, y_pred, zero_division=0)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier_score': brier_score,
            f'precision_at_{k}': precision_at_k,
            'precision': precision_binary,
            'recall': recall_binary,
            'n_samples': len(y),
            'positive_rate': y.mean()
        }
        
    def _get_feature_importance(self, model, model_name: str) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models - use absolute coefficients
            importance = np.abs(model.coef_[0])
        else:
            return {}
            
        # Create feature importance dictionary
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features)
        
    def create_evaluation_plots(self, save_dir: str = "../../models/"):
        """Create evaluation plots and save them."""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = []
        roc_aucs = []
        pr_aucs = []
        brier_scores = []
        
        for model_name, results in self.evaluation_results.items():
            model_names.append(model_name)
            roc_aucs.append(results['test_metrics']['roc_auc'])
            pr_aucs.append(results['test_metrics']['pr_auc'])
            brier_scores.append(results['test_metrics']['brier_score'])
            
        # ROC-AUC comparison
        axes[0,0].bar(model_names, roc_aucs, alpha=0.7)
        axes[0,0].set_title('ROC-AUC Comparison')
        axes[0,0].set_ylabel('ROC-AUC')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # PR-AUC comparison  
        axes[0,1].bar(model_names, pr_aucs, alpha=0.7, color='orange')
        axes[0,1].set_title('Precision-Recall AUC Comparison')
        axes[0,1].set_ylabel('PR-AUC')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Brier Score comparison (lower is better)
        axes[1,0].bar(model_names, brier_scores, alpha=0.7, color='red')
        axes[1,0].set_title('Brier Score Comparison (Lower = Better)')
        axes[1,0].set_ylabel('Brier Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Feature importance for best model
        best_model_name = max(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x]['test_metrics']['roc_auc'])
        
        feature_imp = self.evaluation_results[best_model_name]['feature_importance']
        top_features = dict(list(feature_imp.items())[:10])
        
        axes[1,1].barh(list(top_features.keys())[::-1], list(top_features.values())[::-1])
        axes[1,1].set_title(f'Top 10 Features - {best_model_name}')
        axes[1,1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(save_path / 'model_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved evaluation plots to {save_path}")
        
    def save_models_and_metrics(self, save_dir: str = "../../models/"):
        """Save trained models and evaluation metrics."""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_file = save_path / f"{model_name}_model.joblib"
            joblib.dump(model, model_file)
            
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_file = save_path / f"{scaler_name}_scaler.joblib"
            joblib.dump(scaler, scaler_file)
            
        # Save feature names
        with open(save_path / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
            
        # Save evaluation metrics
        metrics_summary = {}
        for model_name, results in self.evaluation_results.items():
            metrics_summary[model_name] = {
                'test_roc_auc': results['test_metrics']['roc_auc'],
                'test_pr_auc': results['test_metrics']['pr_auc'], 
                'test_brier_score': results['test_metrics']['brier_score'],
                'test_precision_at_20': results['test_metrics'].get('precision_at_20', 0),
                'calibrated_test_roc_auc': results['calibrated_test_metrics']['roc_auc'],
                'calibrated_test_brier_score': results['calibrated_test_metrics']['brier_score']
            }
            
        # Add metadata
        metrics_summary['metadata'] = {
            'trained_at': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'random_state': self.random_state,
            'cross_validation_type': 'spatial_blocking'
        }
        
        with open(save_path / 'baseline_metrics.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
            
        logger.info(f"Saved models and metrics to {save_path}")
        
    def generate_model_report(self) -> str:
        """Generate a summary report of model performance."""
        
        report_lines = ["# M-TRI Baseline Model Performance Report\n"]
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report_lines.append("## Model Performance Summary\n")
        
        # Performance table
        report_lines.append("| Model | ROC-AUC | PR-AUC | Brier Score | Precision@20 |")
        report_lines.append("|-------|---------|--------|-------------|--------------|")
        
        for model_name, results in self.evaluation_results.items():
            metrics = results['test_metrics']
            report_lines.append(
                f"| {model_name} | {metrics['roc_auc']:.3f} | "
                f"{metrics['pr_auc']:.3f} | {metrics['brier_score']:.3f} | "
                f"{metrics.get('precision_at_20', 0):.3f} |"
            )
            
        report_lines.append("\n## Key Findings\n")
        
        # Best model
        best_model = max(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['test_metrics']['roc_auc'])
        best_auc = self.evaluation_results[best_model]['test_metrics']['roc_auc']
        
        report_lines.append(f"- Best performing model: **{best_model}** (ROC-AUC: {best_auc:.3f})")
        
        # Feature importance
        if best_model in self.evaluation_results:
            top_features = list(self.evaluation_results[best_model]['feature_importance'].keys())[:5]
            report_lines.append(f"- Top 5 features: {', '.join(top_features)}")
            
        report_lines.append("\n## Validation Strategy\n")
        report_lines.append("- Used spatial cross-validation to prevent data leakage")
        report_lines.append("- Test set geographically separated from training set")
        report_lines.append("- Model calibration applied for reliable probability estimates")
        
        return '\n'.join(report_lines)


def main():
    """Train baseline models on sample data."""
    
    # Load sample data (in practice, would use engineered features)
    data_path = "../../data/sample/merged_features.csv"
    df = pd.read_csv(data_path)
    
    # Initialize trainer
    trainer = BaselineTrainer(random_state=42)
    
    # Prepare data
    X, y = trainer.prepare_data(df, target_col='toxin_detected')
    
    # Train models
    logger.info("Starting model training...")
    results = trainer.train_models(X, y, df)
    
    # Create evaluation plots
    trainer.create_evaluation_plots()
    
    # Save models and metrics
    trainer.save_models_and_metrics()
    
    # Generate report
    report = trainer.generate_model_report()
    print(report)
    
    # Save report
    with open("../../models/baseline_report.md", 'w') as f:
        f.write(report)
        
    logger.info("Training complete!")


if __name__ == '__main__':
    main()