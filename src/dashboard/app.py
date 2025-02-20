"""
Streamlit dashboard for M-TRI toxin prediction system.
Interactive web interface for exploring pond data and model predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import requests
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Any
import logging

# Configure page
st.set_page_config(
    page_title="M-TRI Dashboard",
    page_icon=":ocean:",
    layout="wide",
    initial_sidebar_state="expanded"
)# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.high-risk {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
}
.medium-risk {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
}
.low-risk {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "demo-token"  # In production, use proper authentication

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_pond_data():
    """Load pond data from API or local file."""
    try:
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.get(f"{API_BASE_URL}/ponds", headers=headers, timeout=10)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
    except Exception as e:
        st.warning(f"API not available, using local data: {e}")
        
    # Fallback to local data
    try:
        return pd.read_csv("../../data/sample/merged_features.csv")
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_pond_rankings(target_date: date, top_n: int = 50):
    """Get pond rankings from API."""
    try:
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        params = {"date": target_date.isoformat(), "top": top_n}
        response = requests.get(f"{API_BASE_URL}/rankings", 
                              headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.warning(f"API not available for rankings: {e}")
        
    return None

def get_pond_prediction(pond_id: str, target_date: date):
    """Get prediction for specific pond."""
    try:
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        payload = {"pond_id": pond_id, "date": target_date.isoformat()}
        response = requests.post(f"{API_BASE_URL}/predict", 
                               headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.warning(f"Could not get prediction for {pond_id}: {e}")
        
    return None

def create_risk_color(risk_score: float) -> str:
    """Get color based on risk score."""
    if risk_score >= 0.7:
        return "#f44336"  # Red
    elif risk_score >= 0.4:
        return "#ff9800"  # Orange  
    else:
        return "#4caf50"  # Green

def create_nj_map(pond_data: pd.DataFrame, rankings_data: Dict = None):
    """Create interactive map of New Jersey ponds."""
    
    if pond_data.empty:
        return None
        
    # Center map on New Jersey
    center_lat = pond_data['lat'].mean() if 'lat' in pond_data.columns else 40.7128
    center_lon = pond_data['lon'].mean() if 'lon' in pond_data.columns else -74.0060
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
    
    # Add pond markers
    for idx, pond in pond_data.iterrows():
        # Get risk info from rankings if available
        risk_score = 0.0
        priority_score = 0.0
        
        if rankings_data and 'rankings' in rankings_data:
            pond_ranking = next((r for r in rankings_data['rankings'] 
                               if r['pond_id'] == pond.get('pond_id', '')), None)
            if pond_ranking:
                risk_score = pond_ranking.get('p_toxin', 0.0)
                priority_score = pond_ranking.get('priority_score', 0.0)
        
        # Determine marker properties
        color = create_risk_color(risk_score)
        
        if risk_score >= 0.7:
            icon = 'exclamation-sign'
        elif risk_score >= 0.4:
            icon = 'warning-sign'
        else:
            icon = 'ok-sign'
            
        # Create popup content
        popup_content = f"""
        <b>Pond {pond.get('pond_id', 'Unknown')}</b><br>
        Toxin Risk: {risk_score:.1%}<br>
        Priority Score: {priority_score:.1f}<br>
        Location: {pond.get('lat', 0):.4f}, {pond.get('lon', 0):.4f}<br>
        """
        
        if 'pond_area_m2' in pond:
            popup_content += f"Area: {pond['pond_area_m2']:,.0f} m²<br>"
            
        folium.Marker(
            location=[pond.get('lat', 0), pond.get('lon', 0)],
            popup=popup_content,
            icon=folium.Icon(color='red' if risk_score >= 0.7 
                           else 'orange' if risk_score >= 0.4 else 'green', 
                           icon=icon)
        ).add_to(m)
    
    return m

def create_time_series_plot(pond_data: pd.DataFrame, pond_id: str):
    """Create time series plot for pond features."""
    
    if pond_data.empty or 'date' not in pond_data.columns:
        return None
        
    pond_subset = pond_data[pond_data['pond_id'] == pond_id].copy()
    
    if pond_subset.empty:
        return None
        
    pond_subset['date'] = pd.to_datetime(pond_subset['date'])
    pond_subset = pond_subset.sort_values('date')
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Chlorophyll Levels', 'NDVI (Vegetation)', 
                       'Nutrient Levels', 'Toxin Detection'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Chlorophyll proxy
    if 'chlorophyll_proxy_14d' in pond_subset.columns:
        fig.add_trace(
            go.Scatter(x=pond_subset['date'], y=pond_subset['chlorophyll_proxy_14d'],
                      name='Chlorophyll', line=dict(color='green')),
            row=1, col=1
        )
    
    # NDVI
    if 'ndvi_mean_14d' in pond_subset.columns:
        fig.add_trace(
            go.Scatter(x=pond_subset['date'], y=pond_subset['ndvi_mean_14d'],
                      name='NDVI', line=dict(color='darkgreen')),
            row=1, col=2
        )
    
    # Nutrients
    if 'phosphate_mean_7d' in pond_subset.columns:
        fig.add_trace(
            go.Scatter(x=pond_subset['date'], y=pond_subset['phosphate_mean_7d'],
                      name='Phosphate', line=dict(color='blue')),
            row=2, col=1
        )
    
    if 'nitrate_mean_7d' in pond_subset.columns:
        fig.add_trace(
            go.Scatter(x=pond_subset['date'], y=pond_subset['nitrate_mean_7d'],
                      name='Nitrate', line=dict(color='red')),
            row=2, col=1, secondary_y=True
        )
    
    # Toxin detection
    if 'toxin_detected' in pond_subset.columns:
        fig.add_trace(
            go.Scatter(x=pond_subset['date'], y=pond_subset['toxin_detected'],
                      mode='markers', name='Toxin Detected',
                      marker=dict(color='red', size=10)),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text=f"Pond {pond_id} - Historical Trends")
    
    return fig

def create_feature_importance_plot(explanation_data: List[Dict]):
    """Create feature importance plot from SHAP-like explanations."""
    
    if not explanation_data:
        return None
        
    # Extract top features
    features = [item['feature'] for item in explanation_data[:10]]
    contributions = [item['contribution'] for item in explanation_data[:10]]
    values = [item['value'] for item in explanation_data[:10]]
    
    # Create horizontal bar plot
    fig = go.Figure(data=go.Bar(
        x=contributions,
        y=features,
        orientation='h',
        marker_color=['red' if c > 0 else 'blue' for c in contributions],
        text=[f"{v:.3f}" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Feature Contributions to Toxin Risk",
        xaxis_title="Contribution",
        yaxis_title="Features",
        height=400
    )
    
    return fig

# Main dashboard
def main():
    """Main dashboard application."""
    
    # Header
    # Main title with custom styling
    st.markdown('<h1 class="main-header">M-TRI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Microbial Toxin-Risk Index** - Real-time harmful algal bloom prediction for New Jersey waterbodies")
    
    # Sidebar
    st.sidebar.header("Controls")
    
    # Date selection
    selected_date = st.sidebar.date_input(
        "Analysis Date",
        value=date.today(),
        min_value=date(2024, 1, 1),
        max_value=date.today()
    )
    
    # Load data
    with st.spinner("Loading pond data..."):
        pond_data = load_pond_data()
        
    if pond_data.empty:
        st.error("No pond data available. Please check data sources.")
        return
        
    # Get rankings
    with st.spinner("Loading risk rankings..."):
        rankings_data = get_pond_rankings(selected_date)
        
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Map Overview", "Rankings", "Pond Detail", "Analytics"])
    
    with tab1:
        st.header("Pond Risk Overview Map")
        
        col1, col2, col3 = st.columns(3)
        
        # Summary metrics
        total_ponds = len(pond_data['pond_id'].unique()) if 'pond_id' in pond_data.columns else 0
        high_risk_count = 0
        avg_risk = 0.0
        
        if rankings_data and 'rankings' in rankings_data:
            high_risk_count = len([r for r in rankings_data['rankings'] if r.get('p_toxin', 0) > 0.5])
            risks = [r.get('p_toxin', 0) for r in rankings_data['rankings']]
            avg_risk = np.mean(risks) if risks else 0.0
            
        with col1:
            st.metric("Total Ponds", total_ponds)
        with col2:
            st.metric("High Risk", high_risk_count, delta=None)
        with col3:
            st.metric("Avg Risk", f"{avg_risk:.1%}")
            
        # Create and display map
        map_obj = create_nj_map(pond_data, rankings_data)
        if map_obj:
            map_data = st_folium(map_obj, width=1000, height=500)
        
    with tab2:
        st.header("Pond Risk Rankings")
        
        if rankings_data and 'rankings' in rankings_data:
            rankings_df = pd.DataFrame(rankings_data['rankings'])
            
            # Risk level filter
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                options=["All", "High (>70%)", "Medium (40-70%)", "Low (<40%)"]
            )
            
            # Apply filter
            if risk_filter == "High (>70%)":
                rankings_df = rankings_df[rankings_df['p_toxin'] > 0.7]
            elif risk_filter == "Medium (40-70%)":
                rankings_df = rankings_df[(rankings_df['p_toxin'] >= 0.4) & (rankings_df['p_toxin'] <= 0.7)]
            elif risk_filter == "Low (<40%)":
                rankings_df = rankings_df[rankings_df['p_toxin'] < 0.4]
                
            # Display rankings table
            st.subheader(f"Top {len(rankings_df)} Ponds by Priority")
            
            # Format for display
            display_df = rankings_df.copy()
            display_df['Toxin Risk'] = display_df['p_toxin'].apply(lambda x: f"{x:.1%}")
            display_df['Priority Score'] = display_df['priority_score'].round(1)
            display_df['Spread Risk'] = display_df['spread_risk_30d'].apply(lambda x: f"{x:.1%}")
            
            # Color code rows by risk
            def highlight_risk(row):
                risk = row['p_toxin']
                if risk >= 0.7:
                    return ['background-color: #ffebee'] * len(row)
                elif risk >= 0.4:
                    return ['background-color: #fff3e0'] * len(row)
                else:
                    return ['background-color: #e8f5e8'] * len(row)
                    
            st.dataframe(
                display_df[['pond_id', 'Toxin Risk', 'Priority Score', 'Spread Risk', 'latitude', 'longitude']]
                .style.apply(highlight_risk, axis=1),
                use_container_width=True
            )
            
            # Download rankings
            csv = rankings_df.to_csv(index=False)
            st.download_button(
                label="Download Rankings CSV",
                data=csv,
                file_name=f"pond_rankings_{selected_date}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("Rankings data not available. Check API connection.")
    
    with tab3:
        st.header("Individual Pond Analysis")
        
        # Pond selection
        if 'pond_id' in pond_data.columns:
            available_ponds = sorted(pond_data['pond_id'].unique())
            selected_pond = st.selectbox("Select Pond", available_ponds)
            
            if selected_pond:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Get pond prediction
                    prediction = get_pond_prediction(selected_pond, selected_date)
                    
                    if prediction:
                        st.subheader(f"Pond {selected_pond} - Risk Assessment")
                        
                        # Risk metrics
                        risk_cols = st.columns(4)
                        with risk_cols[0]:
                            st.metric("Toxin Risk", f"{prediction['p_toxin']:.1%}")
                        with risk_cols[1]:
                            st.metric("Priority Score", f"{prediction['priority_score']:.1f}")
                        with risk_cols[2]:
                            st.metric("Spread Risk", f"{prediction['spread_risk_30d']:.1%}")
                        with risk_cols[3]:
                            confidence = prediction.get('confidence_interval', [0, 0])
                            st.metric("Confidence", f"±{(confidence[1] - confidence[0])/2:.2f}")
                            
                        # Feature importance
                        if prediction.get('explanation'):
                            st.subheader("Risk Factor Analysis")
                            importance_plot = create_feature_importance_plot(prediction['explanation'])
                            if importance_plot:
                                st.plotly_chart(importance_plot, use_container_width=True)
                                
                        # Evidence links
                        if prediction.get('evidence_links'):
                            st.subheader("Supporting Evidence")
                            for link in prediction['evidence_links']:
                                st.info(f"Info: {link['description']}")
                                
                    else:
                        st.warning(f"Could not get prediction for pond {selected_pond}")
                        
                with col2:
                    st.subheader("Pond Information")
                    pond_info = pond_data[pond_data['pond_id'] == selected_pond].iloc[0]
                    
                    st.info(f"""
                    **Location:** {pond_info.get('lat', 'N/A'):.4f}, {pond_info.get('lon', 'N/A'):.4f}
                    
                    **Area:** {pond_info.get('pond_area_m2', 'N/A'):,.0f} m²
                    
                    **Recent Observations:** Available
                    """)
                
                # Time series plot
                st.subheader("Historical Trends")
                ts_plot = create_time_series_plot(pond_data, selected_pond)
                if ts_plot:
                    st.plotly_chart(ts_plot, use_container_width=True)
                else:
                    st.info("Historical data not available for time series analysis")
                    
    with tab4:
        st.header("System Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Coverage")
            
            if not pond_data.empty:
                # Temporal coverage
                if 'date' in pond_data.columns:
                    pond_data['date'] = pd.to_datetime(pond_data['date'])
                    date_range = pond_data['date'].max() - pond_data['date'].min()
                    st.metric("Data Span", f"{date_range.days} days")
                    
                # Feature completeness
                if pond_data.select_dtypes(include=[np.number]).columns.any():
                    completeness = (1 - pond_data.select_dtypes(include=[np.number]).isnull().mean()).mean()
                    st.metric("Data Completeness", f"{completeness:.1%}")
                    
        with col2:
            st.subheader("Model Performance")
            
            # Mock model metrics (in practice, load from saved metrics)
            st.metric("ROC-AUC", "0.78")
            st.metric("Precision@20", "0.45")
            st.metric("Brier Score", "0.18")
            
        # Feature distribution
        if not pond_data.empty:
            st.subheader("Feature Distributions")
            
            # Select numeric columns for analysis
            numeric_cols = pond_data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_features = st.multiselect(
                    "Select features to visualize",
                    options=numeric_cols,
                    default=numeric_cols[:4]  # First 4 features
                )
                
                if selected_features:
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=selected_features[:4]
                    )
                    
                    for i, feature in enumerate(selected_features[:4]):
                        row = (i // 2) + 1
                        col = (i % 2) + 1
                        
                        fig.add_trace(
                            go.Histogram(x=pond_data[feature], name=feature, nbinsx=20),
                            row=row, col=col
                        )
                        
                    fig.update_layout(height=600, title_text="Feature Distributions")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**M-TRI Dashboard** | Built with Streamlit | Data updated: " + 
                datetime.now().strftime("%Y-%m-%d %H:%M"))

if __name__ == "__main__":
    main()