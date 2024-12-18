"""
FastAPI backend for M-TRI toxin prediction system.
Provides REST API endpoints for pond data, predictions, and rankings.
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime, date
from pathlib import Path
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="M-TRI API",
    description="Microbial Toxin-Risk Index - Pond toxin prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PondInfo(BaseModel):
    pond_id: str
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    area_m2: Optional[float] = Field(None, gt=0, description="Pond area in square meters")
    name: Optional[str] = None

class PredictionRequest(BaseModel):
    pond_id: str
    date: date
    features: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    pond_id: str
    date: date
    p_toxin: float = Field(..., ge=0, le=1, description="Probability of toxin detection")
    spread_risk_30d: float = Field(..., ge=0, le=1, description="Risk of spread in 30 days") 
    priority_score: float = Field(..., ge=0, le=100, description="Priority score for testing")
    confidence_interval: List[float] = Field(..., description="95% confidence interval for p_toxin")
    explanation: List[Dict[str, Any]] = Field(..., description="SHAP-like feature explanations")
    evidence_links: List[Dict[str, str]] = Field(..., description="Links to supporting evidence")

class RankingRequest(BaseModel):
    date: date
    top: int = Field(20, ge=1, le=500, description="Number of top ponds to return")
    min_priority: float = Field(0.1, ge=0, le=1, description="Minimum priority threshold")

class RankingResponse(BaseModel):
    date: date
    rankings: List[Dict[str, Any]]
    total_ponds_evaluated: int
    high_risk_count: int

# Global variables for loaded models and data
models = {}
scalers = {}
feature_names = []
pond_data = pd.DataFrame()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API token (simplified for demo)."""
    # In production, implement proper token validation
    expected_token = os.getenv("API_SECRET_KEY", "demo-token")
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

@app.on_event("startup")
async def load_models():
    """Load trained models and supporting data on startup."""
    global models, scalers, feature_names, pond_data
    
    logger.info("Loading models and data...")
    
    try:
        # Load models
        model_path = Path("../../models")
        
        if (model_path / "logistic_model.joblib").exists():
            models["logistic"] = joblib.load(model_path / "logistic_model.joblib")
            logger.info("Loaded logistic regression model")
            
        if (model_path / "random_forest_model.joblib").exists():
            models["random_forest"] = joblib.load(model_path / "random_forest_model.joblib")
            logger.info("Loaded random forest model")
            
        # Load scalers
        if (model_path / "logistic_scaler.joblib").exists():
            scalers["logistic"] = joblib.load(model_path / "logistic_scaler.joblib")
            
        # Load feature names
        if (model_path / "feature_names.json").exists():
            with open(model_path / "feature_names.json") as f:
                feature_names = json.load(f)
                
        # Load pond data
        pond_data_path = Path("../../data/sample/merged_features.csv")
        if pond_data_path.exists():
            pond_data = pd.read_csv(pond_data_path)
            pond_data['date'] = pd.to_datetime(pond_data['date'])
            logger.info(f"Loaded pond data: {len(pond_data)} observations")
            
        logger.info("Model loading complete")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Continue with empty models for demo purposes

def prepare_features(pond_id: str, target_date: date, 
                    custom_features: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Prepare features for a specific pond and date."""
    
    if custom_features:
        # Use provided features
        feature_vector = [custom_features.get(name, 0.0) for name in feature_names]
        return np.array(feature_vector).reshape(1, -1)
    
    # Get features from pond data
    pond_obs = pond_data[
        (pond_data['pond_id'] == pond_id) & 
        (pond_data['date'].dt.date == target_date)
    ]
    
    if pond_obs.empty:
        # Find nearest date observation
        pond_all = pond_data[pond_data['pond_id'] == pond_id]
        if pond_all.empty:
            raise HTTPException(status_code=404, detail=f"No data found for pond {pond_id}")
            
        # Use most recent observation
        pond_obs = pond_all.loc[[pond_all['date'].idxmax()]]
        
    # Extract features
    feature_vector = []
    for name in feature_names:
        if name in pond_obs.columns:
            value = pond_obs[name].iloc[0]
            # Handle missing values
            if pd.isna(value):
                value = 0.0
        else:
            value = 0.0
        feature_vector.append(value)
        
    return np.array(feature_vector).reshape(1, -1)

def calculate_priority_score(p_toxin: float, pond_info: Dict[str, Any]) -> float:
    """Calculate priority score for testing based on multiple factors."""
    
    # Base priority from toxin probability
    priority = p_toxin * 100
    
    # Adjust for pond characteristics
    if 'pond_area_m2' in pond_info and pond_info['pond_area_m2']:
        # Larger ponds get slight priority boost
        area_factor = min(1.2, 1 + pond_info['pond_area_m2'] / 100000)
        priority *= area_factor
        
    # Adjust for recent observations (mock implementation)
    # In practice, would consider recency of last test, accessibility, etc.
    
    return min(100, priority)

def generate_explanation(model, features: np.ndarray, pond_id: str) -> List[Dict[str, Any]]:
    """Generate SHAP-like explanation for prediction."""
    
    # Simplified explanation - in practice would use actual SHAP
    explanations = []
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based model
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear model
        importances = np.abs(model.coef_[0])
    else:
        return explanations
        
    # Get top contributing features
    feature_contributions = list(zip(feature_names, importances, features[0]))
    feature_contributions.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature_name, importance, feature_value) in enumerate(feature_contributions[:10]):
        explanations.append({
            "feature": feature_name,
            "value": float(feature_value),
            "contribution": float(importance),
            "description": _get_feature_description(feature_name)
        })
        
    return explanations

def _get_feature_description(feature_name: str) -> str:
    """Get human-readable description of feature."""
    
    descriptions = {
        'chlorophyll_proxy_14d': 'Average chlorophyll levels (14-day)',
        'ndvi_mean_14d': 'Vegetation index (14-day average)',
        'phosphate_mean_7d': 'Phosphate concentration (7-day average)',
        'nitrate_mean_7d': 'Nitrate concentration (7-day average)',
        'turbidity_latest': 'Water turbidity (most recent)',
        'pond_area_m2': 'Pond surface area',
        'edna_mcy_detected': 'Toxin genes detected in DNA samples',
        'bloom_season': 'Summer bloom season indicator'
    }
    
    return descriptions.get(feature_name, feature_name.replace('_', ' ').title())

@app.get("/", response_model=Dict[str, str])
async def root():
    """API health check and information."""
    return {
        "message": "M-TRI API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy"
    }

@app.get("/ponds", response_model=List[PondInfo])
async def get_ponds():
    """Get list of all available ponds."""
    
    if pond_data.empty:
        return []
        
    # Get unique ponds with their info
    pond_list = []
    for pond_id in pond_data['pond_id'].unique():
        pond_info = pond_data[pond_data['pond_id'] == pond_id].iloc[0]
        pond_list.append(PondInfo(
            pond_id=pond_id,
            latitude=pond_info['lat'],
            longitude=pond_info['lon'],
            area_m2=pond_info.get('pond_area_m2'),
            name=f"Pond {pond_id}"
        ))
        
    return pond_list

@app.get("/ponds/{pond_id}", response_model=Dict[str, Any])
async def get_pond_info(pond_id: str):
    """Get detailed information for a specific pond."""
    
    pond_obs = pond_data[pond_data['pond_id'] == pond_id]
    
    if pond_obs.empty:
        raise HTTPException(status_code=404, detail=f"Pond {pond_id} not found")
        
    # Get latest observation
    latest = pond_obs.loc[pond_obs['date'].idxmax()]
    
    # Calculate summary statistics
    summary = {
        "pond_id": pond_id,
        "latitude": latest['lat'],
        "longitude": latest['lon'],
        "area_m2": latest.get('pond_area_m2'),
        "total_observations": len(pond_obs),
        "date_range": {
            "start": pond_obs['date'].min().date().isoformat(),
            "end": pond_obs['date'].max().date().isoformat()
        },
        "toxin_detection_rate": pond_obs.get('toxin_detected', pd.Series()).mean(),
        "latest_observation": latest['date'].date().isoformat(),
        "recent_features": {}
    }
    
    # Add recent feature values
    key_features = ['chlorophyll_proxy_14d', 'ndvi_mean_14d', 'phosphate_mean_7d', 'nitrate_mean_7d']
    for feature in key_features:
        if feature in latest:
            summary["recent_features"][feature] = latest[feature]
            
    return summary

@app.post("/predict", response_model=PredictionResponse)
async def predict_toxin_risk(request: PredictionRequest, 
                           token: str = Depends(verify_token)):
    """Predict toxin risk for a specific pond and date."""
    
    if not models:
        raise HTTPException(status_code=503, detail="Models not available")
        
    try:
        # Prepare features
        features = prepare_features(request.pond_id, request.date, request.features)
        
        # Use best available model (prefer calibrated versions)
        model_name = "logistic"  # Default
        if f"{model_name}_calibrated" in models:
            model = models[f"{model_name}_calibrated"]
        elif model_name in models:
            model = models[model_name]
        else:
            raise HTTPException(status_code=503, detail="No trained models available")
            
        # Scale features if needed
        if model_name in scalers:
            features_scaled = scalers[model_name].transform(features)
        else:
            features_scaled = features
            
        # Make prediction
        p_toxin = float(model.predict_proba(features_scaled)[0, 1])
        
        # Calculate confidence interval (simplified)
        ci_width = 0.1  # Mock confidence interval
        confidence_interval = [
            max(0, p_toxin - ci_width),
            min(1, p_toxin + ci_width)
        ]
        
        # Calculate spread risk (mock implementation)
        spread_risk_30d = min(1.0, p_toxin * 1.2)  # Simplified spread model
        
        # Get pond info for priority calculation
        pond_info = {}
        if not pond_data.empty:
            pond_obs = pond_data[pond_data['pond_id'] == request.pond_id]
            if not pond_obs.empty:
                latest_obs = pond_obs.iloc[-1]
                pond_info = {
                    'pond_area_m2': latest_obs.get('pond_area_m2'),
                    'lat': latest_obs.get('lat'),
                    'lon': latest_obs.get('lon')
                }
        
        # Calculate priority score
        priority_score = calculate_priority_score(p_toxin, pond_info)
        
        # Generate explanation
        explanation = generate_explanation(model, features_scaled, request.pond_id)
        
        # Create evidence links
        evidence_links = [
            {
                "type": "satellite_imagery",
                "description": "Recent satellite imagery",
                "url": f"/evidence/satellite/{request.pond_id}/{request.date}"
            },
            {
                "type": "water_chemistry", 
                "description": "Water quality measurements",
                "url": f"/evidence/chemistry/{request.pond_id}/{request.date}"
            }
        ]
        
        return PredictionResponse(
            pond_id=request.pond_id,
            date=request.date,
            p_toxin=p_toxin,
            spread_risk_30d=spread_risk_30d,
            priority_score=priority_score,
            confidence_interval=confidence_interval,
            explanation=explanation,
            evidence_links=evidence_links
        )
        
    except Exception as e:
        logger.error(f"Prediction error for {request.pond_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/rankings", response_model=RankingResponse)
async def get_pond_rankings(date: date = None, 
                           top: int = 20,
                           min_priority: float = 0.1,
                           token: str = Depends(verify_token)):
    """Get ranked list of ponds by toxin risk priority."""
    
    if date is None:
        date = datetime.now().date()
        
    if not models or pond_data.empty:
        raise HTTPException(status_code=503, detail="Service not available")
        
    rankings = []
    
    # Get predictions for all ponds
    for pond_id in pond_data['pond_id'].unique():
        try:
            # Make prediction
            prediction = await predict_toxin_risk(
                PredictionRequest(pond_id=pond_id, date=date),
                token="demo-token"  # Internal call
            )
            
            if prediction.priority_score >= min_priority * 100:
                # Get pond location
                pond_obs = pond_data[pond_data['pond_id'] == pond_id].iloc[0]
                
                rankings.append({
                    "pond_id": pond_id,
                    "priority_score": prediction.priority_score,
                    "p_toxin": prediction.p_toxin,
                    "spread_risk_30d": prediction.spread_risk_30d,
                    "latitude": pond_obs['lat'],
                    "longitude": pond_obs['lon'],
                    "area_m2": pond_obs.get('pond_area_m2'),
                    "top_risk_factors": [exp["feature"] for exp in prediction.explanation[:3]]
                })
                
        except Exception as e:
            logger.warning(f"Failed to rank pond {pond_id}: {e}")
            continue
            
    # Sort by priority score
    rankings.sort(key=lambda x: x["priority_score"], reverse=True)
    
    # Limit to top N
    rankings = rankings[:top]
    
    # Count high risk ponds
    high_risk_count = len([r for r in rankings if r["p_toxin"] > 0.5])
    
    return RankingResponse(
        date=date,
        rankings=rankings,
        total_ponds_evaluated=len(pond_data['pond_id'].unique()),
        high_risk_count=high_risk_count
    )

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models),
        "scalers_loaded": len(scalers),
        "features_count": len(feature_names),
        "pond_data_loaded": not pond_data.empty,
        "pond_count": len(pond_data['pond_id'].unique()) if not pond_data.empty else 0
    }
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)