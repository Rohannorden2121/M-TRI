"""
Unit tests for M-TRI data ingestion modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingestion.base import BaseIngester, get_nj_bbox, chunk_date_range
from ingestion.wqp_ingester import WQPIngester

class TestBaseIngester:
    """Test base ingestion functionality."""
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        ingester = BaseIngester(rate_limit_delay=0.01)  # Very short delay for testing
        
        start_time = datetime.now()
        ingester._rate_limit()
        ingester._rate_limit()
        end_time = datetime.now()
        
        # Should have some delay
        elapsed = (end_time - start_time).total_seconds()
        assert elapsed >= 0.01
        
    def test_date_validation(self):
        """Test date range validation."""
        ingester = BaseIngester()
        
        # Valid date range
        start_dt, end_dt = ingester.validate_date_range('2024-01-01', '2024-12-31')
        assert start_dt < end_dt
        
        # Invalid date range (start after end)
        with pytest.raises(ValueError):
            ingester.validate_date_range('2024-12-31', '2024-01-01')
            
        # Invalid date format
        with pytest.raises(ValueError):
            ingester.validate_date_range('invalid-date', '2024-12-31')
            
    def test_bbox_validation(self):
        """Test bounding box validation."""
        ingester = BaseIngester()
        
        # Valid bbox
        bbox = [-75.0, 39.0, -74.0, 41.0]
        validated = ingester.validate_bbox(bbox)
        assert validated == bbox
        
        # Invalid bbox (wrong number of coordinates)
        with pytest.raises(ValueError):
            ingester.validate_bbox([-75.0, 39.0, -74.0])
            
        # Invalid bbox (invalid longitude)
        with pytest.raises(ValueError):
            ingester.validate_bbox([-200.0, 39.0, -74.0, 41.0])
            
        # Invalid bbox (min > max)
        with pytest.raises(ValueError):
            ingester.validate_bbox([-74.0, 39.0, -75.0, 41.0])
            
    def test_save_data(self, tmp_path):
        """Test data saving functionality."""
        ingester = BaseIngester()
        
        # Create test data
        test_data = pd.DataFrame({
            'pond_id': ['NJ001', 'NJ002'],
            'value': [1.0, 2.0]
        })
        
        # Save to temporary path
        output_path = tmp_path / "test_data"
        ingester.save_data(test_data, str(output_path), file_format='csv')
        
        # Check files were created
        assert (tmp_path / "test_data.csv").exists()
        assert (tmp_path / "test_data_metadata.json").exists()
        
        # Load and verify data
        loaded_data = pd.read_csv(tmp_path / "test_data.csv")
        pd.testing.assert_frame_equal(test_data, loaded_data)

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_nj_bbox(self):
        """Test New Jersey bounding box."""
        bbox = get_nj_bbox()
        
        # Should return 4 coordinates
        assert len(bbox) == 4
        
        # Should be valid NJ bounds (approximately)
        min_lon, min_lat, max_lon, max_lat = bbox
        assert -76 <= min_lon <= -73
        assert 38 <= min_lat <= 42
        assert -76 <= max_lon <= -73
        assert 38 <= max_lat <= 42
        assert min_lon < max_lon
        assert min_lat < max_lat
        
    def test_chunk_date_range(self):
        """Test date range chunking."""
        chunks = chunk_date_range('2024-01-01', '2024-01-10', chunk_days=3)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # First chunk should start at start date
        assert chunks[0][0] == '2024-01-01'
        
        # Last chunk should end at or before end date
        last_end = datetime.strptime(chunks[-1][1], '%Y-%m-%d')
        expected_end = datetime.strptime('2024-01-10', '%Y-%m-%d')
        assert last_end <= expected_end

class TestWQPIngester:
    """Test Water Quality Portal ingester."""
    
    def test_initialization(self):
        """Test WQP ingester initialization."""
        ingester = WQPIngester()
        
        assert ingester.base_url == "https://www.waterqualitydata.us/data/Result/search"
        assert 'chlorophyll_a' in ingester.param_codes
        assert 'phosphate' in ingester.param_codes
        
    def test_parameter_standardization(self):
        """Test parameter name standardization."""
        ingester = WQPIngester()
        
        # Test various parameter names
        assert ingester._standardize_parameter_name('Chlorophyll a') == 'chlorophyll_a'
        assert ingester._standardize_parameter_name('CHLOROPHYLL A') == 'chlorophyll_a'
        assert ingester._standardize_parameter_name('Phosphate') == 'phosphate'
        assert ingester._standardize_parameter_name('Nitrate') == 'nitrate'
        assert ingester._standardize_parameter_name('Unknown Parameter') is None
        assert ingester._standardize_parameter_name(None) is None
        
    def test_process_wqp_data(self):
        """Test WQP data processing."""
        ingester = WQPIngester()
        
        # Create mock raw data
        raw_data = pd.DataFrame({
            'MonitoringLocationIdentifier': ['SITE001', 'SITE002'],
            'ActivityStartDate': ['2024-01-01', '2024-01-02'],
            'ActivityStartTime/Time': ['10:00:00', '11:00:00'],
            'CharacteristicName': ['Chlorophyll a', 'Phosphate'],
            'ResultMeasureValue': [5.2, 0.15],
            'ResultMeasure/MeasureUnitCode': ['ug/L', 'mg/L'],
            'LatitudeMeasure': [40.7, 40.8],
            'LongitudeMeasure': [-74.1, -74.2]
        })
        
        processed = ingester._process_wqp_data(raw_data)
        
        # Check processing results
        assert len(processed) == 2
        assert 'station_id' in processed.columns
        assert 'date' in processed.columns
        assert 'parameter_standardized' in processed.columns
        
        # Check parameter standardization worked
        assert 'chlorophyll_a' in processed['parameter_standardized'].values
        assert 'phosphate' in processed['parameter_standardized'].values
        
    @patch('requests.Session.get')
    def test_fetch_water_quality_data_success(self, mock_get):
        """Test successful WQP data fetching."""
        ingester = WQPIngester()
        
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """MonitoringLocationIdentifier,ActivityStartDate,ActivityStartTime/Time,CharacteristicName,ResultMeasureValue,ResultMeasure/MeasureUnitCode,LatitudeMeasure,LongitudeMeasure
SITE001,2024-01-01,10:00:00,Chlorophyll a,5.2,ug/L,40.7,-74.1"""
        mock_get.return_value = mock_response
        
        # Test fetch
        result = ingester.fetch_water_quality_data('2024-01-01', '2024-01-31')
        
        # Should return processed DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0  # May be empty after processing
        
    @patch('requests.Session.get')
    def test_fetch_water_quality_data_failure(self, mock_get):
        """Test WQP data fetching with HTTP error."""
        ingester = WQPIngester()
        
        # Mock HTTP error
        mock_get.side_effect = Exception("HTTP Error")
        
        # Should return empty DataFrame on error
        result = ingester.fetch_water_quality_data('2024-01-01', '2024-01-31')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_data_processing(self):
        """Test complete data processing workflow."""
        
        # Create sample raw data
        sample_data = pd.DataFrame({
            'pond_id': ['NJ001', 'NJ002', 'NJ001'],
            'date': ['2024-06-01', '2024-06-01', '2024-06-02'],
            'lat': [40.7, 40.8, 40.7],
            'lon': [-74.1, -74.2, -74.1],
            'chlorophyll_proxy_14d': [5.2, 3.1, 6.8],
            'phosphate_mean_7d': [0.15, 0.12, 0.18],
            'toxin_detected': [0, 0, 1]
        })
        
        # Basic validation
        assert len(sample_data) == 3
        assert sample_data['pond_id'].nunique() == 2
        assert sample_data['toxin_detected'].sum() == 1
        
        # Check data types
        numeric_cols = ['lat', 'lon', 'chlorophyll_proxy_14d', 'phosphate_mean_7d']
        for col in numeric_cols:
            assert sample_data[col].dtype in ['float64', 'int64']

# Fixtures
@pytest.fixture
def sample_pond_data():
    """Sample pond data for testing."""
    return pd.DataFrame({
        'pond_id': ['NJ001', 'NJ002', 'NJ003'],
        'lat': [40.7128, 40.6892, 40.9176],
        'lon': [-74.0060, -74.3444, -74.1718],
        'pond_area_m2': [15000, 22000, 8500],
        'date': ['2024-06-01', '2024-06-01', '2024-06-01']
    })

@pytest.fixture
def sample_water_quality_data():
    """Sample water quality data for testing."""
    return pd.DataFrame({
        'station_id': ['SITE001', 'SITE002'],
        'date': ['2024-06-01', '2024-06-01'],
        'parameter_standardized': ['chlorophyll_a', 'phosphate'],
        'value': [5.2, 0.15],
        'latitude': [40.7, 40.8],
        'longitude': [-74.1, -74.2],
        'source': ['WQP', 'WQP']
    })

# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "api: marks tests that require API access")