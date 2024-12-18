"""
Tests for M-TRI model training and evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.train_baseline import BaselineTrainer
    from features.engineering import FeatureEngineering
except ImportError:
    # Skip if sklearn not available
    pytest.skip("scikit-learn not available", allow_module_level=True)

class TestBaselineTrainer:
    """Test baseline model training functionality."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'pond_id': [f'NJ{i:03d}' for i in range(n_samples)],
            'date': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
            'lat': np.random.uniform(40.5, 41.0, n_samples),
            'lon': np.random.uniform(-74.5, -74.0, n_samples),
            'chlorophyll_proxy_14d': np.random.exponential(5, n_samples),
            'phosphate_mean_7d': np.random.exponential(0.1, n_samples),
            'nitrate_mean_7d': np.random.exponential(1.0, n_samples),
            'ndvi_mean_14d': np.random.uniform(0.2, 0.8, n_samples),
            'pond_area_m2': np.random.uniform(5000, 50000, n_samples),
            'turbidity_latest': np.random.exponential(10, n_samples)
        })
        
        # Create correlated target variable
        toxin_prob = (
            0.1 + 
            0.3 * (data['chlorophyll_proxy_14d'] > 8).astype(int) +
            0.2 * (data['phosphate_mean_7d'] > 0.15).astype(int) +
            0.2 * (data['nitrate_mean_7d'] > 1.5).astype(int)
        )
        data['toxin_detected'] = np.random.binomial(1, toxin_prob, n_samples)
        
        return data
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = BaselineTrainer(random_state=42)
        
        assert trainer.random_state == 42
        assert len(trainer.model_configs) > 0
        assert 'logistic' in trainer.model_configs
        assert 'random_forest' in trainer.model_configs
        
    def test_prepare_data(self, sample_training_data):
        """Test data preparation."""
        trainer = BaselineTrainer()
        
        X, y = trainer.prepare_data(sample_training_data)
        
        # Check shapes
        assert len(X) == len(y)
        assert len(X) == len(sample_training_data)
        
        # Check that excluded columns were removed
        excluded_cols = ['pond_id', 'date', 'lat', 'lon', 'toxin_detected']
        for col in excluded_cols:
            assert col not in X.columns
            
        # Check feature names stored
        assert len(trainer.feature_names) == len(X.columns)
        
    def test_spatial_train_test_split(self, sample_training_data):
        """Test spatial cross-validation split."""
        trainer = BaselineTrainer(random_state=42)
        
        X, y = trainer.prepare_data(sample_training_data)
        
        X_train, X_test, y_train, y_test = trainer.spatial_train_test_split(
            sample_training_data, X, y, test_size=0.2
        )
        
        # Check split sizes
        total_size = len(X)
        assert len(X_train) + len(X_test) == total_size
        assert len(y_train) + len(y_test) == total_size
        
        # Test set should be roughly 20% (may vary due to spatial clustering)
        test_ratio = len(X_test) / total_size
        assert 0.1 <= test_ratio <= 0.3  # Allow some variation
        
    def test_model_training(self, sample_training_data):
        """Test model training workflow."""
        trainer = BaselineTrainer(random_state=42)
        
        X, y = trainer.prepare_data(sample_training_data)
        
        # Train models
        results = trainer.train_models(X, y, sample_training_data)
        
        # Check results structure
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for model_name, model_results in results.items():
            # Check required metrics
            assert 'train_metrics' in model_results
            assert 'test_metrics' in model_results
            assert 'feature_importance' in model_results
            
            # Check metric values are reasonable
            test_metrics = model_results['test_metrics']
            assert 0 <= test_metrics['roc_auc'] <= 1
            assert 0 <= test_metrics['pr_auc'] <= 1
            assert 0 <= test_metrics['brier_score'] <= 1
            
    def test_model_evaluation_metrics(self, sample_training_data):
        """Test evaluation metric calculations."""
        trainer = BaselineTrainer(random_state=42)
        
        # Create mock model for testing
        class MockModel:
            def predict_proba(self, X):
                # Return random probabilities that somewhat correlate with first feature
                probs_positive = 0.3 + 0.4 * (X[:, 0] > np.median(X[:, 0]))
                probs_negative = 1 - probs_positive
                return np.column_stack([probs_negative, probs_positive])
                
            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] > 0.5).astype(int)
        
        model = MockModel()
        
        # Create test data
        X_test = np.random.randn(50, 5)
        y_test = np.random.binomial(1, 0.3, 50)  # 30% positive rate
        
        # Evaluate model
        metrics = trainer._evaluate_model(model, X_test, y_test, 'test')
        
        # Check all expected metrics are present
        expected_metrics = ['roc_auc', 'pr_auc', 'brier_score', 'precision', 'recall']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            
    def test_feature_importance_extraction(self, sample_training_data):
        """Test feature importance extraction."""
        trainer = BaselineTrainer()
        
        # Prepare data to set feature names
        X, y = trainer.prepare_data(sample_training_data)
        
        # Mock model with feature importance
        class MockTreeModel:
            feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.05])
            
        model = MockTreeModel()
        
        # Test extraction
        importance_dict = trainer._get_feature_importance(model, 'mock_model')
        
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) <= len(trainer.feature_names)
        
        # Check that importances sum to reasonable value
        total_importance = sum(importance_dict.values())
        assert total_importance > 0
        
    def test_save_and_load_models(self, sample_training_data, tmp_path):
        """Test model saving and loading."""
        trainer = BaselineTrainer(random_state=42)
        
        X, y = trainer.prepare_data(sample_training_data)
        results = trainer.train_models(X, y, sample_training_data)
        
        # Save models
        trainer.save_models_and_metrics(save_dir=str(tmp_path))
        
        # Check files were created
        assert (tmp_path / 'baseline_metrics.json').exists()
        assert (tmp_path / 'feature_names.json').exists()
        
        # Check that at least one model file exists
        model_files = list(tmp_path.glob('*_model.joblib'))
        assert len(model_files) > 0

class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def sample_pond_data(self):
        """Sample pond data."""
        return pd.DataFrame({
            'pond_id': ['NJ001', 'NJ002', 'NJ001'],
            'date': ['2024-06-01', '2024-06-01', '2024-06-15'],
            'lat': [40.7, 40.8, 40.7],
            'lon': [-74.1, -74.2, -74.1],
            'pond_area_m2': [15000, 22000, 15000]
        })
        
    def test_feature_engineer_initialization(self):
        """Test feature engineering initialization."""
        fe = FeatureEngineering()
        
        assert hasattr(fe, 'feature_config')
        assert 'remote_sensing' in fe.feature_config
        assert 'chemistry' in fe.feature_config
        
    def test_temporal_features(self, sample_pond_data):
        """Test temporal feature creation."""
        fe = FeatureEngineering()
        
        # Add temporal features
        result = fe._add_temporal_features(sample_pond_data)
        
        # Check new columns were added
        expected_cols = ['day_of_year', 'month', 'bloom_season', 'day_of_year_sin', 'day_of_year_cos']
        for col in expected_cols:
            assert col in result.columns
            
        # Check value ranges
        assert result['day_of_year'].min() >= 1
        assert result['day_of_year'].max() <= 366
        assert result['month'].min() >= 1
        assert result['month'].max() <= 12
        assert result['bloom_season'].isin([0, 1]).all()
        
    def test_contextual_features(self, sample_pond_data):
        """Test contextual feature creation."""
        fe = FeatureEngineering()
        
        result = fe._add_contextual_features(sample_pond_data)
        
        # Check land use features were added
        land_use_cols = ['agriculture_pct', 'urban_pct', 'forest_pct']
        for col in land_use_cols:
            assert col in result.columns
            assert result[col].min() >= 0
            assert result[col].max() <= 100
            
    def test_feature_validation(self, sample_pond_data):
        """Test feature validation."""
        fe = FeatureEngineering()
        
        # Create data with some issues
        problem_data = sample_pond_data.copy()
        problem_data['inf_feature'] = [np.inf, -np.inf, 1.0]
        problem_data['constant_feature'] = 5.0
        problem_data['missing_feature'] = [1.0, np.nan, np.nan]
        
        # Validate
        clean_data, issues = fe.validate_features(problem_data)
        
        # Check that issues were detected
        assert len(issues) > 0
        
        # Check that infinite values were handled
        assert not np.isinf(clean_data.select_dtypes(include=[np.number])).any().any()
        
        # Check that constant features were removed
        assert 'constant_feature' not in clean_data.columns

class TestModelIntegration:
    """Integration tests for complete modeling workflow."""
    
    def test_end_to_end_training_workflow(self):
        """Test complete training workflow."""
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 50  # Small for fast testing
        
        data = pd.DataFrame({
            'pond_id': [f'NJ{i:03d}' for i in range(n_samples)],
            'date': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
            'lat': np.random.uniform(40.5, 41.0, n_samples),
            'lon': np.random.uniform(-74.5, -74.0, n_samples),
            'pond_area_m2': np.random.uniform(5000, 50000, n_samples),
            'chlorophyll_proxy_14d': np.random.exponential(5, n_samples),
            'phosphate_mean_7d': np.random.exponential(0.1, n_samples),
            'ndvi_mean_14d': np.random.uniform(0.2, 0.8, n_samples)
        })
        
        # Create realistic target
        risk_score = (
            (data['chlorophyll_proxy_14d'] > 7).astype(float) * 0.4 +
            (data['phosphate_mean_7d'] > 0.12).astype(float) * 0.3 +
            np.random.uniform(0, 0.3, n_samples)
        )
        data['toxin_detected'] = (risk_score > 0.5).astype(int)
        
        # Feature engineering
        fe = FeatureEngineering()
        
        # Create base pond data
        pond_data = data[['pond_id', 'date', 'lat', 'lon', 'pond_area_m2']].copy()
        
        # Add features
        features = fe.create_features(pond_data=pond_data)
        
        # Add target back
        features['toxin_detected'] = data['toxin_detected']
        
        # Train model
        trainer = BaselineTrainer(random_state=42)
        X, y = trainer.prepare_data(features)
        
        # Should complete without errors
        results = trainer.train_models(X, y, features)
        
        # Basic result validation
        assert len(results) > 0
        for model_name, model_results in results.items():
            assert 'test_metrics' in model_results
            test_auc = model_results['test_metrics']['roc_auc']
            assert 0 <= test_auc <= 1

# Test markers
pytestmark = pytest.mark.models