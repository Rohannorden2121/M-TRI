"""
USGS NWIS (National Water Information System) data ingestion.
Fetches streamflow, gauge height, and hydrologic data.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from .base import BaseIngester, get_nj_bbox, chunk_date_range

logger = logging.getLogger(__name__)

class USGSIngester(BaseIngester):
    """Ingester for USGS NWIS hydrologic data."""
    
    def __init__(self):
        super().__init__(rate_limit_delay=0.2)
        self.base_url = "https://waterservices.usgs.gov/nwis"
        
        # Parameter codes for hydrologic data
        self.param_codes = {
            'streamflow': '00060',      # Discharge, cubic feet per second
            'gauge_height': '00065',    # Gauge height, feet
            'temperature': '00010',     # Temperature, water, degrees Celsius
            'ph': '00400',             # pH, standard units
            'dissolved_oxygen': '00300', # Dissolved oxygen, mg/L
            'turbidity': '63680'       # Turbidity, FNU
        }
        
    def fetch_site_info(self, bbox: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Get information about USGS monitoring sites in the area.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            
        Returns:
            DataFrame with site information
        """
        if bbox is None:
            bbox = get_nj_bbox()
            
        self.validate_bbox(bbox)
        
        # Site service URL
        url = f"{self.base_url}/site/"
        
        params = {
            'format': 'rdb',
            'bBox': ','.join(map(str, bbox)),
            'outputDataTypeCd': 'dv',  # Daily values
            'siteStatus': 'active',
            'hasDataTypeCd': 'dv'
        }
        
        try:
            response = self._make_request(url, params=params)
            
            # Parse USGS RDB format (tab-separated, with metadata rows)
            lines = response.text.strip().split('\n')
            
            # Find header row (starts with 'agency_cd')
            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith('agency_cd'):
                    header_idx = i
                    break
                    
            if header_idx is None:
                logger.error("Could not find header in USGS response")
                return pd.DataFrame()
                
            # Skip the format description row after header
            data_start = header_idx + 2
            
            # Read data
            from io import StringIO
            data_text = '\n'.join([lines[header_idx]] + lines[data_start:])
            
            sites = pd.read_csv(StringIO(data_text), sep='\t')
            
            # Clean up column names and data
            sites.columns = [col.strip() for col in sites.columns]
            
            logger.info(f"Found {len(sites)} USGS sites in bounding box")
            return sites
            
        except Exception as e:
            logger.error(f"Failed to fetch USGS site info: {e}")
            return pd.DataFrame()
            
    def fetch_daily_values(self, site_codes: List[str],
                          start_date: str, end_date: str,
                          param_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch daily values for specified sites and parameters.
        
        Args:
            site_codes: List of USGS site codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            param_codes: List of parameter codes to fetch
            
        Returns:
            DataFrame with daily values
        """
        logger.info(f"Fetching USGS daily values for {len(site_codes)} sites")
        
        self.validate_date_range(start_date, end_date)
        
        if param_codes is None:
            param_codes = list(self.param_codes.values())
            
        # URL for daily values service
        url = f"{self.base_url}/dv/"
        
        all_data = []
        
        # Process sites in chunks (USGS has limits on request size)
        site_chunks = [site_codes[i:i+20] for i in range(0, len(site_codes), 20)]
        
        for site_chunk in site_chunks:
            params = {
                'format': 'rdb',
                'sites': ','.join(site_chunk),
                'startDT': start_date,
                'endDT': end_date,
                'parameterCd': ','.join(param_codes),
                'statCd': '00003'  # Mean daily value
            }
            
            try:
                response = self._make_request(url, params=params)
                
                # Parse RDB format
                chunk_data = self._parse_usgs_rdb(response.text)
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                    logger.info(f"Fetched {len(chunk_data)} records for {len(site_chunk)} sites")
                    
            except Exception as e:
                logger.error(f"Failed to fetch data for site chunk: {e}")
                continue
                
        if not all_data:
            logger.warning("No USGS data retrieved")
            return pd.DataFrame()
            
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total USGS records: {len(combined_data)}")
        
        return self._process_usgs_data(combined_data)
        
    def _parse_usgs_rdb(self, rdb_text: str) -> pd.DataFrame:
        """Parse USGS RDB format data."""
        lines = rdb_text.strip().split('\n')
        
        # Find data header
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('agency_cd') and 'site_no' in line:
                header_idx = i
                break
                
        if header_idx is None:
            return pd.DataFrame()
            
        # Skip format row after header
        data_start = header_idx + 2
        
        if data_start >= len(lines):
            return pd.DataFrame()
            
        # Read data
        from io import StringIO
        data_text = '\n'.join([lines[header_idx]] + lines[data_start:])
        
        try:
            df = pd.read_csv(StringIO(data_text), sep='\t')
            return df
        except Exception as e:
            logger.error(f"Failed to parse USGS RDB data: {e}")
            return pd.DataFrame()
            
    def _process_usgs_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize USGS data."""
        if raw_data.empty:
            return raw_data
            
        logger.info("Processing USGS data...")
        
        # Standard columns that should be present
        expected_cols = ['agency_cd', 'site_no', 'datetime', 'tz_cd']
        
        # Check for required columns
        missing_cols = [col for col in expected_cols if col not in raw_data.columns]
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
            
        # Find value columns (end with _00003 for mean daily)
        value_cols = [col for col in raw_data.columns if col.endswith('_00003')]
        
        if not value_cols:
            logger.warning("No data value columns found")
            return pd.DataFrame()
            
        # Melt data to long format
        id_cols = ['agency_cd', 'site_no', 'datetime', 'tz_cd']
        id_cols = [col for col in id_cols if col in raw_data.columns]
        
        melted = raw_data.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='parameter_stat',
            value_name='value'
        )
        
        # Extract parameter code from column name
        melted['parameter_code'] = melted['parameter_stat'].str.extract(r'(\d+)_00003')
        
        # Convert data types
        melted['datetime'] = pd.to_datetime(melted['datetime'], errors='coerce')
        melted['value'] = pd.to_numeric(melted['value'], errors='coerce')
        
        # Remove missing values
        melted = melted.dropna(subset=['datetime', 'value'])
        
        # Map parameter codes to names
        param_name_map = {v: k for k, v in self.param_codes.items()}
        melted['parameter_name'] = melted['parameter_code'].map(param_name_map)
        
        # Clean up column names
        processed = melted.rename(columns={
            'site_no': 'station_id',
            'datetime': 'date',
            'value': 'measurement_value'
        })
        
        # Add metadata
        processed['source'] = 'USGS_NWIS'
        processed['ingested_at'] = pd.Timestamp.now()
        
        # Select final columns
        final_cols = ['station_id', 'date', 'parameter_code', 'parameter_name', 
                     'measurement_value', 'source', 'ingested_at']
        processed = processed[final_cols]
        
        logger.info(f"Processed {len(processed)} USGS records")
        return processed
        
    def find_nearest_stations(self, pond_locations: pd.DataFrame, 
                             max_distance_km: float = 50) -> pd.DataFrame:
        """
        Find the nearest USGS stations to each pond.
        
        Args:
            pond_locations: DataFrame with pond coordinates
            max_distance_km: Maximum search distance
            
        Returns:
            DataFrame mapping ponds to nearest stations
        """
        # Get all available sites
        sites = self.fetch_site_info()
        
        if sites.empty:
            logger.warning("No USGS sites found")
            return pd.DataFrame()
            
        # Calculate distances between ponds and stations
        from scipy.spatial.distance import cdist
        import numpy as np
        
        # Extract coordinates
        pond_coords = pond_locations[['latitude', 'longitude']].values
        site_coords = sites[['dec_lat_va', 'dec_long_va']].values
        
        # Calculate distances (approximate km using degree differences)
        distances = cdist(pond_coords, site_coords, metric='euclidean')
        distances_km = distances * 111  # Rough conversion to km
        
        # Find nearest station for each pond
        nearest_stations = []
        
        for i, pond_row in pond_locations.iterrows():
            pond_distances = distances_km[i]
            min_dist_idx = np.argmin(pond_distances)
            min_distance = pond_distances[min_dist_idx]
            
            if min_distance <= max_distance_km:
                nearest_site = sites.iloc[min_dist_idx]
                nearest_stations.append({
                    'pond_id': pond_row['pond_id'],
                    'nearest_station_id': nearest_site['site_no'],
                    'station_name': nearest_site.get('station_nm', 'Unknown'),
                    'distance_km': min_distance,
                    'station_lat': nearest_site['dec_lat_va'],
                    'station_lon': nearest_site['dec_long_va']
                })
                
        result_df = pd.DataFrame(nearest_stations)
        logger.info(f"Found {len(result_df)} pond-station pairs within {max_distance_km} km")
        
        return result_df


def main():
    """Example usage of USGS ingester."""
    
    ingester = USGSIngester()
    
    # Get site information
    sites = ingester.fetch_site_info()
    print(f"Found {len(sites)} USGS sites")
    
    if not sites.empty:
        # Fetch data for first 5 sites
        sample_sites = sites['site_no'].head(5).tolist()
        
        data = ingester.fetch_daily_values(
            site_codes=sample_sites,
            start_date='2024-06-01', 
            end_date='2024-08-31'
        )
        
        if not data.empty:
            output_path = "../../data/raw/usgs_data"
            ingester.save_data(data, output_path)
            print(f"Saved {len(data)} USGS records")


if __name__ == '__main__':
    main()