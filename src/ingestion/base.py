"""
Base data ingestion utilities for M-TRI project.
Handles common functionality like authentication, rate limiting, and error handling.
"""

import os
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseIngester:
    """Base class for all data ingestion scripts."""
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize base ingester.
        
        Args:
            rate_limit_delay: Seconds to wait between API calls
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
        
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None, timeout: int = 30) -> requests.Response:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            url: Request URL
            params: Query parameters
            headers: HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: On HTTP errors
        """
        self._rate_limit()
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
            
    def validate_date_range(self, start_date: str, end_date: str) -> tuple:
        """
        Validate and parse date range.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Tuple of parsed datetime objects
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_dt > end_dt:
                raise ValueError("Start date must be before end date")
                
            if end_dt > datetime.now():
                logger.warning("End date is in the future")
                
            return start_dt, end_dt
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            raise
            
    def validate_bbox(self, bbox: List[float]) -> List[float]:
        """
        Validate bounding box coordinates.
        
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            
        Returns:
            Validated bbox
        """
        if len(bbox) != 4:
            raise ValueError("Bbox must have 4 coordinates: [min_lon, min_lat, max_lon, max_lat]")
            
        min_lon, min_lat, max_lon, max_lat = bbox
        
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
            
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
            
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("Invalid bbox: min values must be less than max values")
            
        return bbox
        
    def save_data(self, data: pd.DataFrame, output_path: str, 
                  file_format: str = 'parquet') -> None:
        """
        Save data to file with metadata.
        
        Args:
            data: DataFrame to save
            output_path: Output file path (without extension)
            file_format: File format ('parquet', 'csv', 'feather')
        """
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'n_records': len(data),
            'columns': list(data.columns),
            'ingestion_source': self.__class__.__name__
        }
        
        # Save data
        if file_format == 'parquet':
            data.to_parquet(f"{output_path}.parquet", index=False)
        elif file_format == 'csv':
            data.to_csv(f"{output_path}.csv", index=False)
        elif file_format == 'feather':
            data.to_feather(f"{output_path}.feather")
        else:
            raise ValueError(f"Unsupported format: {file_format}")
            
        # Save metadata
        import json
        with open(f"{output_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved {len(data)} records to {output_path}.{file_format}")


def get_nj_bbox() -> List[float]:
    """Get bounding box for New Jersey."""
    # New Jersey approximate bounds
    return [-75.56, 38.93, -73.89, 41.36]


def chunk_date_range(start_date: str, end_date: str, 
                    chunk_days: int = 30) -> List[tuple]:
    """
    Split date range into smaller chunks for API requests.
    
    Args:
        start_date: Start date string
        end_date: End date string  
        chunk_days: Days per chunk
        
    Returns:
        List of (start_date, end_date) tuples
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    chunks = []
    current_start = start_dt
    
    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)
        chunks.append((
            current_start.strftime('%Y-%m-%d'),
            current_end.strftime('%Y-%m-%d')
        ))
        current_start = current_end + timedelta(days=1)
        
    return chunks