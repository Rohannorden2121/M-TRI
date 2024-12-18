"""
Water Quality Portal (WQP) data ingestion for M-TRI project.
Fetches nutrient data, chlorophyll-a, turbidity, and other water chemistry parameters.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from .base import BaseIngester, get_nj_bbox, chunk_date_range

logger = logging.getLogger(__name__)

class WQPIngester(BaseIngester):
    """Ingester for Water Quality Portal data."""
    
    def __init__(self):
        super().__init__(rate_limit_delay=0.5)  # Be nice to the public API
        self.base_url = "https://www.waterqualitydata.us/data/Result/search"
        
        # Key parameter codes we care about
        self.param_codes = {
            'chlorophyll_a': ['00078', '32211'],  # Chlorophyll a
            'phosphate': ['00665', '00671'],      # Phosphate, Orthophosphate
            'nitrate': ['00618', '71851'],       # Nitrate, Nitrate+Nitrite
            'turbidity': ['00076', '61028'],     # Turbidity
            'temperature': ['00010'],            # Water temperature
            'ph': ['00400'],                     # pH
            'dissolved_oxygen': ['00300']        # Dissolved oxygen
        }
        
    def fetch_water_quality_data(self, start_date: str, end_date: str,
                                bbox: Optional[List[float]] = None,
                                characteristic_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch water quality data from WQP.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            characteristic_names: Specific parameters to fetch
            
        Returns:
            DataFrame with water quality measurements
        """
        logger.info(f"Fetching WQP data from {start_date} to {end_date}")
        
        if bbox is None:
            bbox = get_nj_bbox()
            
        self.validate_date_range(start_date, end_date)
        self.validate_bbox(bbox)
        
        # Get all parameter codes if not specified
        if characteristic_names is None:
            all_codes = []
            for param_list in self.param_codes.values():
                all_codes.extend(param_list)
        else:
            all_codes = []
            for name in characteristic_names:
                if name in self.param_codes:
                    all_codes.extend(self.param_codes[name])
        
        # Build query parameters
        params = {
            'startDateLo': start_date,
            'startDateHi': end_date,
            'bBox': ','.join(map(str, bbox)),
            'pCode': ';'.join(all_codes),
            'mimeType': 'csv',
            'dataProfile': 'resultPhysChem',
            'sorted': 'no'
        }
        
        # Fetch data in chunks if date range is large
        date_chunks = chunk_date_range(start_date, end_date, chunk_days=90)
        all_data = []
        
        for chunk_start, chunk_end in date_chunks:
            chunk_params = params.copy()
            chunk_params['startDateLo'] = chunk_start
            chunk_params['startDateHi'] = chunk_end
            
            try:
                response = self._make_request(self.base_url, params=chunk_params)
                
                # Parse CSV response
                from io import StringIO
                chunk_data = pd.read_csv(StringIO(response.text))
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                    logger.info(f"Fetched {len(chunk_data)} records for {chunk_start} to {chunk_end}")
                else:
                    logger.warning(f"No data for {chunk_start} to {chunk_end}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch chunk {chunk_start} to {chunk_end}: {e}")
                continue
                
        if not all_data:
            logger.warning("No water quality data found")
            return pd.DataFrame()
            
        # Combine all chunks
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total WQP records retrieved: {len(combined_data)}")
        
        return self._process_wqp_data(combined_data)
        
    def _process_wqp_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize WQP data.
        
        Args:
            raw_data: Raw WQP DataFrame
            
        Returns:
            Processed DataFrame
        """
        if raw_data.empty:
            return raw_data
            
        logger.info("Processing WQP data...")
        
        # Key columns we need
        required_columns = [
            'MonitoringLocationIdentifier',
            'ActivityStartDate',
            'ActivityStartTime/Time',
            'CharacteristicName', 
            'ResultMeasureValue',
            'ResultMeasure/MeasureUnitCode',
            'LatitudeMeasure',
            'LongitudeMeasure'
        ]
        
        # Check if required columns exist
        missing_cols = [col for col in required_columns if col not in raw_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
            
        # Select and rename columns
        processed = raw_data[required_columns].copy()
        processed.rename(columns={
            'MonitoringLocationIdentifier': 'station_id',
            'ActivityStartDate': 'date',
            'ActivityStartTime/Time': 'time',
            'CharacteristicName': 'parameter',
            'ResultMeasureValue': 'value',
            'ResultMeasure/MeasureUnitCode': 'unit',
            'LatitudeMeasure': 'latitude',
            'LongitudeMeasure': 'longitude'
        }, inplace=True)
        
        # Convert data types
        processed['date'] = pd.to_datetime(processed['date'], errors='coerce')
        processed['value'] = pd.to_numeric(processed['value'], errors='coerce')
        processed['latitude'] = pd.to_numeric(processed['latitude'], errors='coerce')
        processed['longitude'] = pd.to_numeric(processed['longitude'], errors='coerce')
        
        # Remove records with missing critical data
        initial_count = len(processed)
        processed = processed.dropna(subset=['date', 'value', 'latitude', 'longitude'])
        logger.info(f"Removed {initial_count - len(processed)} records with missing critical data")
        
        # Standardize parameter names
        processed['parameter_standardized'] = processed['parameter'].map(self._standardize_parameter_name)
        
        # Filter out parameters we don't recognize
        processed = processed[processed['parameter_standardized'].notna()]
        
        # Add metadata
        processed['ingested_at'] = pd.Timestamp.now()
        processed['source'] = 'WQP'
        
        logger.info(f"Processed {len(processed)} WQP records")
        return processed
        
    def _standardize_parameter_name(self, param_name: str) -> Optional[str]:
        """Standardize parameter names to our internal schema."""
        if pd.isna(param_name):
            return None
            
        param_lower = param_name.lower()
        
        # Mapping rules
        if 'chlorophyll' in param_lower and 'a' in param_lower:
            return 'chlorophyll_a'
        elif 'phosphate' in param_lower or 'phosphorus' in param_lower:
            return 'phosphate'
        elif 'nitrate' in param_lower:
            return 'nitrate'
        elif 'turbidity' in param_lower:
            return 'turbidity'
        elif 'temperature' in param_lower:
            return 'temperature'
        elif 'ph' in param_lower:
            return 'ph'
        elif 'oxygen' in param_lower:
            return 'dissolved_oxygen'
        else:
            return None


def main():
    """Example usage of WQP ingester."""
    import os
    
    ingester = WQPIngester()
    
    # Fetch recent data for New Jersey
    data = ingester.fetch_water_quality_data(
        start_date='2024-06-01',
        end_date='2024-08-31'
    )
    
    if not data.empty:
        output_path = "../../data/raw/wqp_data"
        ingester.save_data(data, output_path)
        print(f"Saved {len(data)} WQP records")
    else:
        print("No data retrieved")


if __name__ == '__main__':
    main()