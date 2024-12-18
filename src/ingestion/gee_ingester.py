"""
Google Earth Engine (GEE) satellite imagery ingestion for M-TRI project.
Fetches Sentinel-2 and Landsat data for vegetation and water quality indices.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from .base import BaseIngester

logger = logging.getLogger(__name__)

class GEEIngester(BaseIngester):
    """Ingester for Google Earth Engine satellite data."""
    
    def __init__(self):
        super().__init__(rate_limit_delay=0.1)  # GEE handles rate limiting internally
        self.ee = None
        self._initialize_ee()
        
    def _initialize_ee(self):
        """Initialize Earth Engine authentication."""
        try:
            import ee
            
            # Try to initialize (will use existing credentials)
            try:
                ee.Initialize()
                logger.info("Earth Engine initialized successfully")
            except Exception:
                # If initialization fails, try authenticating
                logger.info("Authenticating with Earth Engine...")
                ee.Authenticate()
                ee.Initialize()
                logger.info("Earth Engine authenticated and initialized")
                
            self.ee = ee
            
        except ImportError:
            logger.error("Earth Engine API not installed. Install with: pip install earthengine-api")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            raise
            
    def fetch_satellite_data(self, pond_locations: pd.DataFrame,
                           start_date: str, end_date: str,
                           buffer_m: int = 100) -> pd.DataFrame:
        """
        Fetch satellite data for pond locations.
        
        Args:
            pond_locations: DataFrame with 'pond_id', 'latitude', 'longitude' columns
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            buffer_m: Buffer around pond center in meters
            
        Returns:
            DataFrame with satellite-derived features per pond per date
        """
        logger.info(f"Fetching satellite data for {len(pond_locations)} ponds")
        
        if self.ee is None:
            raise RuntimeError("Earth Engine not initialized")
            
        # Convert dates
        start_dt, end_dt = self.validate_date_range(start_date, end_date)
        
        results = []
        
        for idx, pond in pond_locations.iterrows():
            try:
                pond_data = self._fetch_pond_timeseries(
                    pond['pond_id'],
                    pond['latitude'], 
                    pond['longitude'],
                    start_date, end_date, buffer_m
                )
                results.extend(pond_data)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(pond_locations)} ponds")
                    
            except Exception as e:
                logger.error(f"Failed to process pond {pond['pond_id']}: {e}")
                continue
                
        if not results:
            logger.warning("No satellite data retrieved")
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        logger.info(f"Retrieved satellite data: {len(df)} pond-date combinations")
        
        return df
        
    def _fetch_pond_timeseries(self, pond_id: str, lat: float, lon: float,
                              start_date: str, end_date: str, buffer_m: int) -> List[Dict]:
        """Fetch time series data for a single pond."""
        ee = self.ee
        
        # Create point geometry with buffer
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(buffer_m)
        
        # Date range for filtering
        date_filter = ee.Filter.date(start_date, end_date)
        
        # Get Sentinel-2 data
        s2_data = self._get_sentinel2_features(area, date_filter)
        
        # Get Landsat data (as backup/additional source)
        landsat_data = self._get_landsat_features(area, date_filter)
        
        # Combine results
        results = []
        
        # Process Sentinel-2 data
        for item in s2_data:
            item['pond_id'] = pond_id
            item['latitude'] = lat
            item['longitude'] = lon
            item['source'] = 'Sentinel-2'
            results.append(item)
            
        # Process Landsat data
        for item in landsat_data:
            item['pond_id'] = pond_id
            item['latitude'] = lat
            item['longitude'] = lon
            item['source'] = 'Landsat'
            results.append(item)
            
        return results
        
    def _get_sentinel2_features(self, area, date_filter) -> List[Dict]:
        """Extract features from Sentinel-2 imagery."""
        ee = self.ee
        
        # Load Sentinel-2 collection
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filter(date_filter)
              .filterBounds(area)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Low cloud cover
              .sort('system:time_start'))
        
        # Function to calculate indices for each image
        def calculate_indices(image):
            # Get bands
            B2 = image.select('B2')  # Blue
            B3 = image.select('B3')  # Green  
            B4 = image.select('B4')  # Red
            B8 = image.select('B8')  # NIR
            B11 = image.select('B11') # SWIR1
            
            # Calculate vegetation indices
            ndvi = B8.subtract(B4).divide(B8.add(B4)).rename('ndvi')
            
            # Enhanced Vegetation Index
            evi = image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {'NIR': B8, 'RED': B4, 'BLUE': B2}
            ).rename('evi')
            
            # Floating Algae Index (for harmful algal blooms)
            fai = image.expression(
                'NIR - (RED + (SWIR1 - RED) * (842 - 665) / (1614 - 665))',
                {'NIR': B8, 'RED': B4, 'SWIR1': B11}
            ).rename('fai')
            
            # Chlorophyll index (red edge based)
            B5 = image.select('B5')  # Red Edge
            chlor_idx = B5.subtract(B4).divide(B5.add(B4)).rename('chlor_index')
            
            # Water-leaving reflectance approximation
            water_refl = B3.divide(10000).rename('water_reflectance')
            
            return image.addBands([ndvi, evi, fai, chlor_idx, water_refl])
        
        # Apply calculations to collection
        s2_processed = s2.map(calculate_indices)
        
        # Reduce to get mean values over the area for each date
        def extract_features(image):
            # Calculate mean values over the area
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=area,
                scale=10,  # 10m resolution for Sentinel-2
                maxPixels=1e9
            )
            
            # Get image date
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            
            return ee.Feature(None, stats.set('date', date))
        
        # Extract features for all images
        features = s2_processed.map(extract_features)
        
        # Convert to list
        try:
            feature_list = features.getInfo()['features']
            
            results = []
            for feat in feature_list:
                props = feat['properties']
                if props.get('ndvi') is not None:  # Only include if we got valid data
                    results.append({
                        'date': props['date'],
                        'ndvi': props.get('ndvi'),
                        'evi': props.get('evi'),
                        'fai': props.get('fai'),
                        'chlorophyll_index': props.get('chlor_index'),
                        'water_reflectance': props.get('water_reflectance'),
                        'cloud_cover': None  # Could add cloud stats here
                    })
                    
            return results
            
        except Exception as e:
            logger.warning(f"Failed to extract Sentinel-2 features: {e}")
            return []
            
    def _get_landsat_features(self, area, date_filter) -> List[Dict]:
        """Extract features from Landsat imagery."""
        ee = self.ee
        
        # Load Landsat 8/9 collection
        landsat = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                   .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
                   .filter(date_filter)
                   .filterBounds(area)
                   .filter(ee.Filter.lt('CLOUD_COVER', 20)))
        
        # Scale factors for Landsat Collection 2
        def apply_scale_factors(image):
            optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
            return image.addBands(optical_bands, None, True)
        
        landsat_scaled = landsat.map(apply_scale_factors)
        
        # Function to calculate indices
        def calculate_landsat_indices(image):
            # Get bands (Landsat band numbers)
            B2 = image.select('SR_B2')  # Blue
            B3 = image.select('SR_B3')  # Green
            B4 = image.select('SR_B4')  # Red  
            B5 = image.select('SR_B5')  # NIR
            B6 = image.select('SR_B6')  # SWIR1
            
            # NDVI
            ndvi = B5.subtract(B4).divide(B5.add(B4)).rename('ndvi_landsat')
            
            # Water indices
            ndwi = B3.subtract(B5).divide(B3.add(B5)).rename('ndwi')
            
            # Modified chlorophyll index
            chlor = B5.subtract(B4).divide(B5.add(B4)).rename('chlor_landsat')
            
            return image.addBands([ndvi, ndwi, chlor])
        
        landsat_processed = landsat_scaled.map(calculate_landsat_indices)
        
        # Extract features similar to Sentinel-2
        def extract_landsat_features(image):
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=area,
                scale=30,  # 30m resolution for Landsat
                maxPixels=1e9
            )
            
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            cloud_cover = image.get('CLOUD_COVER')
            
            return ee.Feature(None, stats.set({'date': date, 'cloud_cover': cloud_cover}))
        
        features = landsat_processed.map(extract_landsat_features)
        
        try:
            feature_list = features.getInfo()['features']
            
            results = []
            for feat in feature_list:
                props = feat['properties']
                if props.get('ndvi_landsat') is not None:
                    results.append({
                        'date': props['date'],
                        'ndvi_landsat': props.get('ndvi_landsat'),
                        'ndwi': props.get('ndwi'),
                        'chlor_landsat': props.get('chlor_landsat'),
                        'cloud_cover_pct': props.get('cloud_cover')
                    })
                    
            return results
            
        except Exception as e:
            logger.warning(f"Failed to extract Landsat features: {e}")
            return []


def main():
    """Example usage of GEE ingester."""
    
    # Example pond locations
    pond_locations = pd.DataFrame({
        'pond_id': ['NJ001', 'NJ002', 'NJ003'],
        'latitude': [40.7128, 40.6892, 40.9176],
        'longitude': [-74.0060, -74.3444, -74.1718]
    })
    
    try:
        ingester = GEEIngester()
        
        # Fetch satellite data
        data = ingester.fetch_satellite_data(
            pond_locations=pond_locations,
            start_date='2024-06-01',
            end_date='2024-08-31',
            buffer_m=50
        )
        
        if not data.empty:
            output_path = "../../data/raw/satellite_data"
            ingester.save_data(data, output_path)
            print(f"Saved {len(data)} satellite observations")
        else:
            print("No satellite data retrieved")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires Google Earth Engine authentication")


if __name__ == '__main__':
    main()