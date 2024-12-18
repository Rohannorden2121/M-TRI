"""
Feature engineering pipeline for M-TRI project.
Transforms raw data into model-ready features for toxin prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Main feature engineering pipeline for pond toxin prediction."""
    
    def __init__(self):
        self.feature_config = {
            # Remote sensing features
            'remote_sensing': {
                'ndvi_windows': [7, 14, 30],  # Days to average over
                'chlor_windows': [7, 14, 30],
                'anomaly_baseline_days': 365  # Days for seasonal baseline
            },
            
            # Water chemistry features  
            'chemistry': {
                'nutrients': ['phosphate', 'nitrate'],
                'aggregation_windows': [7, 14, 30],  # Days for rolling means
                'interpolation_max_gap': 7  # Max days to interpolate
            },
            
            # Hydrologic features
            'hydrology': {
                'flow_windows': [7, 14, 30],
                'precipitation_windows': [7, 14, 30]
            },
            
            # Environmental DNA features
            'edna': {
                'gene_targets': ['mcy', 'mcyA', 'sxt', 'cyl'],
                'abundance_threshold': 0.001  # Minimum relative abundance
            }
        }
        
    def create_features(self, pond_data: pd.DataFrame, 
                       satellite_data: pd.DataFrame = None,
                       chemistry_data: pd.DataFrame = None,
                       hydro_data: pd.DataFrame = None,
                       edna_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create complete feature set for each pond-date combination.
        
        Args:
            pond_data: Base pond information (pond_id, date, lat, lon, area)
            satellite_data: Remote sensing observations  
            chemistry_data: Water quality measurements
            hydro_data: Hydrologic measurements
            edna_data: Environmental DNA detections
            
        Returns:
            DataFrame with engineered features for model training
        """
        logger.info(f"Engineering features for {len(pond_data)} pond observations")
        
        # Start with base pond data
        features = pond_data.copy()
        
        # Ensure date column is datetime
        features['date'] = pd.to_datetime(features['date'])
        
        # Add temporal features
        features = self._add_temporal_features(features)
        
        # Add remote sensing features
        if satellite_data is not None:
            features = self._add_remote_sensing_features(features, satellite_data)
            
        # Add water chemistry features
        if chemistry_data is not None:
            features = self._add_chemistry_features(features, chemistry_data)
            
        # Add hydrologic features
        if hydro_data is not None:
            features = self._add_hydrologic_features(features, hydro_data)
            
        # Add eDNA features
        if edna_data is not None:
            features = self._add_edna_features(features, edna_data)
            
        # Add contextual features
        features = self._add_contextual_features(features)
        
        # Add persistence features
        features = self._add_persistence_features(features)
        
        logger.info(f"Generated {len(features.columns)} features for {len(features)} observations")
        return features
        
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        logger.info("Adding temporal features...")
        
        df = df.copy()
        
        # Basic time components
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Cyclical encoding for seasonal patterns
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Summer bloom season (June-September)
        df['bloom_season'] = ((df['month'] >= 6) & (df['month'] <= 9)).astype(int)
        
        # Days since start of bloom season
        df['days_since_bloom_start'] = (df['date'] - 
                                       pd.to_datetime(df['date'].dt.year.astype(str) + '-06-01')).dt.days
        df['days_since_bloom_start'] = df['days_since_bloom_start'].clip(lower=0)
        
        return df
        
    def _add_remote_sensing_features(self, df: pd.DataFrame, 
                                    satellite_data: pd.DataFrame) -> pd.DataFrame:
        """Add satellite-derived features."""
        logger.info("Adding remote sensing features...")
        
        sat_data = satellite_data.copy()
        sat_data['date'] = pd.to_datetime(sat_data['date'])
        
        # Sort data for time-based calculations
        sat_data = sat_data.sort_values(['pond_id', 'date'])
        
        features_list = []
        
        for pond_id in df['pond_id'].unique():
            pond_df = df[df['pond_id'] == pond_id].copy()
            pond_sat = sat_data[sat_data['pond_id'] == pond_id]
            
            if pond_sat.empty:
                logger.warning(f"No satellite data for pond {pond_id}")
                features_list.append(pond_df)
                continue
                
            # Merge with satellite data using nearest date matching
            merged = pd.merge_asof(
                pond_df.sort_values('date'),
                pond_sat.sort_values('date'),
                on='date',
                by='pond_id',
                direction='nearest',
                tolerance=pd.Timedelta(days=7)  # Max 7 days difference
            )
            
            # Calculate rolling statistics for key indices
            for window in self.feature_config['remote_sensing']['ndvi_windows']:
                if 'ndvi' in pond_sat.columns:
                    # Rolling mean NDVI
                    rolling_ndvi = pond_sat.set_index('date')['ndvi'].rolling(
                        f'{window}D', min_periods=1
                    ).mean()
                    
                    merged[f'ndvi_mean_{window}d'] = merged['date'].map(rolling_ndvi)
                    
                    # NDVI anomaly (vs seasonal baseline)
                    baseline_days = self.feature_config['remote_sensing']['anomaly_baseline_days']
                    seasonal_mean = pond_sat.set_index('date')['ndvi'].rolling(
                        f'{baseline_days}D', min_periods=30
                    ).mean()
                    
                    merged[f'ndvi_anomaly_{window}d'] = (merged[f'ndvi_mean_{window}d'] - 
                                                        merged['date'].map(seasonal_mean))
                                                        
            # Chlorophyll proxy features
            for window in self.feature_config['remote_sensing']['chlor_windows']:
                if 'chlorophyll_index' in pond_sat.columns:
                    rolling_chlor = pond_sat.set_index('date')['chlorophyll_index'].rolling(
                        f'{window}D', min_periods=1
                    ).mean()
                    
                    merged[f'chlorophyll_proxy_{window}d'] = merged['date'].map(rolling_chlor)
                    
            # Water quality indices
            if 'fai' in pond_sat.columns:
                merged['fai_latest'] = merged['fai']
                
            if 'water_reflectance' in pond_sat.columns:
                merged['water_refl_latest'] = merged['water_reflectance']
                
            features_list.append(merged)
            
        # Combine all ponds
        result = pd.concat(features_list, ignore_index=True)
        
        return result
        
    def _add_chemistry_features(self, df: pd.DataFrame, 
                               chemistry_data: pd.DataFrame) -> pd.DataFrame:
        """Add water chemistry features."""
        logger.info("Adding water chemistry features...")
        
        chem_data = chemistry_data.copy()
        chem_data['date'] = pd.to_datetime(chem_data['date'])
        
        # Pivot chemistry data to wide format
        chem_pivot = chem_data.pivot_table(
            index=['station_id', 'date'],
            columns='parameter_standardized',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Map chemistry stations to ponds (using nearest approach)
        features_list = []
        
        for pond_id in df['pond_id'].unique():
            pond_df = df[df['pond_id'] == pond_id].copy()
            
            # For simplicity, use all chemistry data (in practice would map by location)
            pond_chem = chem_pivot.copy()
            
            if pond_chem.empty:
                features_list.append(pond_df)
                continue
                
            # Calculate rolling statistics for nutrients
            for nutrient in self.feature_config['chemistry']['nutrients']:
                if nutrient not in pond_chem.columns:
                    continue
                    
                # Sort by date for time-based operations
                pond_chem_sorted = pond_chem.sort_values('date')
                
                for window in self.feature_config['chemistry']['aggregation_windows']:
                    # Rolling mean
                    rolling_mean = pond_chem_sorted.set_index('date')[nutrient].rolling(
                        f'{window}D', min_periods=1
                    ).mean()
                    
                    # Map to pond dates
                    pond_df[f'{nutrient}_mean_{window}d'] = pond_df['date'].map(rolling_mean)
                    
                    # Latest value (forward fill)
                    latest_vals = pond_chem_sorted.set_index('date')[nutrient].resample('D').ffill()
                    pond_df[f'{nutrient}_latest'] = pond_df['date'].map(latest_vals)
                    
            features_list.append(pond_df)
            
        result = pd.concat(features_list, ignore_index=True)
        return result
        
    def _add_hydrologic_features(self, df: pd.DataFrame, 
                                hydro_data: pd.DataFrame) -> pd.DataFrame:
        """Add hydrologic features."""
        logger.info("Adding hydrologic features...")
        
        # Similar pattern to chemistry features
        # For demo, add simplified hydrologic proxies
        
        df = df.copy()
        
        # Mock hydrologic features (in practice, would use USGS data)
        # Flow proxy based on seasonal patterns
        df['flow_seasonal_mean'] = (
            2.0 + 1.5 * np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        )
        
        # Precipitation anomaly (mock)
        df['precip_anomaly_7d'] = np.random.normal(0, 5, len(df))
        df['precip_anomaly_14d'] = np.random.normal(0, 3, len(df))
        
        return df
        
    def _add_edna_features(self, df: pd.DataFrame, 
                          edna_data: pd.DataFrame) -> pd.DataFrame:
        """Add environmental DNA features."""
        logger.info("Adding eDNA features...")
        
        df = df.copy()
        
        # For demo purposes, create simplified eDNA features
        # In practice, would process actual SRA metadata and gene abundance data
        
        # Binary gene detection features
        for gene in self.feature_config['edna']['gene_targets']:
            df[f'edna_{gene}_detected'] = 0  # Default to not detected
            
        # Relative abundance features  
        df['edna_toxin_gene_abundance'] = 0.0
        
        # Mock some detections based on other features
        if 'chlorophyll_proxy_14d' in df.columns:
            high_chlor_mask = df['chlorophyll_proxy_14d'] > df['chlorophyll_proxy_14d'].quantile(0.7)
            df.loc[high_chlor_mask, 'edna_mcy_detected'] = 1
            df.loc[high_chlor_mask, 'edna_toxin_gene_abundance'] = np.random.exponential(0.002, 
                                                                                        high_chlor_mask.sum())
        
        return df
        
    def _add_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual environmental features."""
        logger.info("Adding contextual features...")
        
        df = df.copy()
        
        # Land use proxies (in practice, would derive from NLCD data)
        # Mock features based on lat/lon
        df['agriculture_pct'] = np.clip(
            30 + 10 * np.sin(df['lat'] * 10) + 5 * np.cos(df['lon'] * 10), 0, 100
        )
        
        df['urban_pct'] = np.clip(
            20 + 15 * np.cos(df['lat'] * 5) + 8 * np.sin(df['lon'] * 8), 0, 100
        )
        
        df['forest_pct'] = np.clip(
            100 - df['agriculture_pct'] - df['urban_pct'], 0, 100
        )
        
        # Distance to roads/development (mock)
        df['distance_to_road_m'] = np.random.exponential(500, len(df))
        
        # Watershed characteristics
        df['upstream_agriculture_pct'] = df['agriculture_pct'] * 1.2  # Slight amplification
        
        return df
        
    def _add_persistence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add persistence and trend features."""
        logger.info("Adding persistence features...")
        
        df = df.copy()
        df = df.sort_values(['pond_id', 'date'])
        
        # Count of recent anomalous conditions
        for pond_id in df['pond_id'].unique():
            pond_mask = df['pond_id'] == pond_id
            pond_df = df[pond_mask]
            
            # Count high chlorophyll days in last 30 days
            if 'chlorophyll_proxy_14d' in df.columns:
                chlor_threshold = df['chlorophyll_proxy_14d'].quantile(0.8)
                high_chlor = (pond_df['chlorophyll_proxy_14d'] > chlor_threshold).astype(int)
                
                high_chlor_30d = high_chlor.rolling(window=30, min_periods=1).sum()
                df.loc[pond_mask, 'high_chlor_days_30d'] = high_chlor_30d.values
                
            # Trend in NDVI (if available)
            if 'ndvi_mean_14d' in df.columns:
                ndvi_trend = pond_df['ndvi_mean_14d'].rolling(window=14).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 3 else 0
                )
                df.loc[pond_mask, 'ndvi_trend_14d'] = ndvi_trend.values
                
        return df
        
    def validate_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate feature quality and return cleaned dataset.
        
        Returns:
            Tuple of (cleaned_dataframe, list_of_issues)
        """
        logger.info("Validating feature quality...")
        
        issues = []
        df_clean = df.copy()
        
        # Check for infinite values
        inf_cols = []
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if np.isinf(df_clean[col]).any():
                inf_cols.append(col)
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                
        if inf_cols:
            issues.append(f"Replaced infinite values in: {inf_cols}")
            
        # Check for high missing data
        missing_pct = df_clean.isnull().sum() / len(df_clean) * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        
        if high_missing:
            issues.append(f"High missing data (>50%) in: {high_missing}")
            
        # Check for constant features
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        constant_cols = []
        
        for col in numeric_cols:
            if df_clean[col].nunique() <= 1:
                constant_cols.append(col)
                
        if constant_cols:
            issues.append(f"Constant features detected: {constant_cols}")
            df_clean = df_clean.drop(columns=constant_cols)
            
        # Check feature ranges
        extreme_ranges = []
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            col_range = df_clean[col].max() - df_clean[col].min()
            if col_range == 0:
                continue
            elif col_range > 1e6:  # Very large range
                extreme_ranges.append(col)
                
        if extreme_ranges:
            issues.append(f"Features with extreme ranges: {extreme_ranges}")
            
        logger.info(f"Feature validation complete. Found {len(issues)} issues.")
        
        return df_clean, issues


def main():
    """Example usage of feature engineering pipeline."""
    
    # Load sample data
    sample_data = pd.read_csv("../../data/sample/merged_features.csv")
    
    # Initialize feature engineer
    fe = FeatureEngineering()
    
    # Create base pond data
    pond_data = sample_data[['pond_id', 'date', 'lat', 'lon', 'pond_area_m2']].copy()
    
    # Mock satellite data
    satellite_data = sample_data[['pond_id', 'date', 'ndvi_mean_14d', 'chlorophyll_proxy_14d']].copy()
    satellite_data.rename(columns={
        'ndvi_mean_14d': 'ndvi',
        'chlorophyll_proxy_14d': 'chlorophyll_index'
    }, inplace=True)
    
    # Create features
    features = fe.create_features(
        pond_data=pond_data,
        satellite_data=satellite_data
    )
    
    # Validate features
    features_clean, issues = fe.validate_features(features)
    
    print(f"Generated {len(features_clean.columns)} features")
    print(f"Validation issues: {len(issues)}")
    for issue in issues:
        print(f"  - {issue}")
        
    # Save results
    output_path = "../../data/processed/engineered_features.csv"
    features_clean.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")


if __name__ == '__main__':
    main()