"""
M-TRI Data Ingestion Package

This package provides ingestion utilities for various data sources:
- Water Quality Portal (WQP) - water chemistry data
- USGS NWIS - hydrologic data and stream flow
- Google Earth Engine - satellite imagery and remote sensing
- NCBI SRA - environmental DNA metadata
"""

from .base import BaseIngester, get_nj_bbox, chunk_date_range
from .wqp_ingester import WQPIngester
from .usgs_ingester import USGSIngester

# Only import GEE if Earth Engine is available
try:
    from .gee_ingester import GEEIngester
except ImportError:
    GEEIngester = None

__all__ = [
    'BaseIngester',
    'WQPIngester', 
    'USGSIngester',
    'GEEIngester',
    'get_nj_bbox',
    'chunk_date_range'
]