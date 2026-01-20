import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
import requests

from preprocess import load_dataset
from utils import inject_css, kpi_row, cache_df
from analytics import zscore_anomalies, moving_average, arima_forecast
from geo import load_geojson, map_state_names, get_geojson_property_key


st.set_page_config(
    page_title="UIDAI Insights Dashboard", 
    page_icon="🇮🇳", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# UIDAI Official Color Scheme CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    body, .stApp { 
        font-family: 'Inter', sans-serif;
        background: #ffffff;
        color: #1e293b;
    }
    
    /* Force all text to be dark */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #1e293b !important;
    }
    
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
    
    .main-header {
        animation: fadeIn 0.8s ease-out;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header h3 {
        color: #e0e7ff !important;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 10px;
    }
    
    .logo-badge {
        display: inline-block;
        background: #f97316;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        color: white !important;
        box-shadow: 0 4px 10px rgba(249, 115, 22, 0.3);
        margin: 5px;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.4);
    }
    
    div[data-testid="stMetric"] label {
        font-weight: 700 !important;
        color: #fff !important;
        text-transform: uppercase;
    }
    
    div[data-testid="stMetric"] > div {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #fff !important;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e293b 100%);
    }
    
    /* Sidebar input fields styling */
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: white !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p {
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #e0e7ff;
        color: #1e3a8a !important;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 10px 10px 0 0;
        border: 1px solid #c7d2fe;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f97316, #ea580c) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.4);
        border: 1px solid #f97316 !important;
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white !important;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 5px 20px rgba(249, 115, 22, 0.3);
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #2563eb, #1e40af);
        color: white !important;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #1e40af, #1e3a8a);
    }
    
    /* Ensure info/success boxes have visible text */
    .stAlert {
        color: #1e293b !important;
    }
    
    .stAlert p {
        color: #1e293b !important;
    }
    
    /* Tab content text */
    .stTabs [data-baseweb="tab-panel"] {
        color: #1e293b !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader {
        color: #1e293b !important;
    }
    
    /* Button text */
    .stButton button {
        color: #1e293b !important;
    }
    
    /* Dataframe text */
    .dataframe {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Title Banner
st.markdown("""
    <div class="main-header">
        <h1> UIDAI Insights Dashboard</h1>
        <h3>
            <span class="logo-badge">📊 Analytics</span>
            <span class="logo-badge">🔍 Anomaly Detection</span>
            <span class="logo-badge">📈 Forecasting</span>
        </h3>
        <h3 style="margin-top: 15px;">Unlocking Societal Trends in Aadhaar Enrolment and Updates</h3>
    </div>
""", unsafe_allow_html=True)

# Google Drive file IDs mapping
GDRIVE_FILES = {
    "enrolment": "1cdQM1TGvlg_ed1PhahwaxQbnHeHW1tbm",
    "biometric": "1FCoAmEtzposGd4VgNyeI_I_jwZWhwzGe",
    "demographic": "1VXQ61eDwrmeVML31g_ubPU__LSPE6FWu"
}

def get_gdrive_download_url(file_id):
    """Convert Google Drive file ID to direct download URL"""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

@st.cache_data
def load_data_from_gdrive(file_id, dataset_type):
    """Load CSV data from Google Drive"""
    try:
        url = get_gdrive_download_url(file_id)
        df = pd.read_csv(url)
        return load_dataset(df, dataset_type, from_dataframe=True)
    except Exception as e:
        st.error(f"Error loading data from Google Drive: {e}")
        return None

# Sidebar
st.sidebar.markdown("## ⚙️ Dashboard Controls")
st.sidebar.markdown("---")

dataset_type = st.sidebar.selectbox(
    "📁 Dataset Type", 
    ["enrolment", "biometric", "demographic"],
    help="Select the type of dataset to analyze"
)

st.sidebar.markdown("### 🎯 Analysis Parameters")
anomaly_threshold = st.sidebar.slider("🚨 Anomaly Sensitivity", 1.5, 4.0, 2.5, 0.1)
ma_window = st.sidebar.slider("📉 Smoothing Window", 3, 30, 7, 1)
forecast_steps = st.sidebar.slider("🔮 Forecast Horizon", 3, 30, 7, 1)

# Load data from Google Drive
file_id = GDRIVE_FILES[dataset_type]
with st.spinner(f"Loading {dataset_type} data from Google Drive..."):
    df = load_data_from_gdrive(file_id, dataset_type)

if df is None:
    st.error(f"❌ Failed to load data from Google Drive")
    st.stop()

df = cache_df(df)
metric_col = {'enrolment': 'total_enrolment', 'biometric': 'total_biometric', 'demographic': 'total_demographic'}[dataset_type]

# KPIs
st.markdown("## 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📝 Total Records", f"{len(df):,}", "Active")
with col2:
    st.metric(f"🎯 {metric_col.replace('_', ' ').title()}", f"{int(df[metric_col].sum()):,}", f"{int(df[metric_col].mean()):,} avg")
with col3:
    st.metric("🗺️ States", f"{df['state'].nunique() if 'state' in df.columns else 0}", "Coverage")
with col4:
    st.metric("📍 Districts", f"{df['district'].nunique() if 'district' in df.columns else 0}", "Zones")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "📊 Geography", "🗺️ Map", "💡 Insights", "📥 Export"])

# Tab 1: Trends
with tab1:
    st.markdown("### 📈 Trends & Anomaly Detection")
    st.info("🎯 **Objective**: Detect unusual patterns for operational insights")
    
    if 'date' in df.columns and df['date'].notna().any():
        ts = df.groupby('date')[metric_col].sum().reset_index().sort_values('date')
        ts['ma'] = moving_average(ts[metric_col], window=ma_window)
        anomalies_idx, z = zscore_anomalies(ts[metric_col], threshold=anomaly_threshold)
        ts['is_anomaly'] = False
        ts.loc[ts.index[anomalies_idx], 'is_anomaly'] = True
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-highlight">🚨 {len(anomalies_idx)} Anomalies</div>', unsafe_allow_html=True)
        with col2:
            peak = ts.loc[ts[metric_col].idxmax(), 'date'].strftime("%d %b %Y")
            st.markdown(f'<div class="metric-highlight">📈 Peak: {peak}</div>', unsafe_allow_html=True)
        with col3:
            avg = int(ts[metric_col].mean())
            st.markdown(f'<div class="metric-highlight">📊 Avg: {avg:,}/day</div>', unsafe_allow_html=True)
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts['date'], y=ts[metric_col], mode='lines+markers', name='Daily', line=dict(color='#2563eb', width=2)))
        fig_ts.add_trace(go.Scatter(x=ts['date'], y=ts['ma'], mode='lines', name=f'{ma_window}-Day MA', line=dict(dash='dash', color='#1e40af')))
        anomalies = ts[ts['is_anomaly']]
        fig_ts.add_trace(go.Scatter(x=anomalies['date'], y=anomalies[metric_col], mode='markers', name='Anomalies', 
                                    marker=dict(color='#f97316', size=12, symbol='diamond', line=dict(color='white', width=2))))
        fig_ts.update_layout(
            title=dict(text="Daily Activity with Anomalies", font=dict(color='#1e293b', size=18)),
            template="plotly_white", 
            height=500, 
            hovermode='x unified',
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#1e293b', size=12),
            xaxis=dict(
                title=dict(text='Date', font=dict(color='#1e293b', size=14)),
                tickfont=dict(color='#1e293b', size=12)
            ),
            yaxis=dict(
                title=dict(text='Activity Count', font=dict(color='#1e293b', size=14)),
                tickfont=dict(color='#1e293b', size=12)
            )
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        if len(anomalies) > 0:
            with st.expander(f"🔍 View {len(anomalies)} Anomalies"):
                anom = anomalies[['date', metric_col]].copy()
                anom['date'] = anom['date'].dt.strftime('%d %b %Y')
                st.dataframe(anom, hide_index=True)
        
        st.markdown("### 🔮 Forecast")
        fc = arima_forecast(ts[['date', metric_col]].rename(columns={metric_col: 'value'}), 'value', steps=forecast_steps)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=ts['date'], y=ts[metric_col], mode='lines', name='History', line=dict(color='#2563eb')))
        fig_fc.add_trace(go.Scatter(x=fc['date'], y=fc['forecast'], mode='lines+markers', name='Forecast', 
                                    line=dict(color='#f97316', width=3, dash='dot')))
        fig_fc.update_layout(
            title=dict(text=f"{forecast_steps}-Day Forecast", font=dict(color='#1e293b', size=18)),
            template="plotly_white", 
            height=400,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#1e293b', size=12),
            xaxis=dict(
                title=dict(text='Date', font=dict(color='#1e293b', size=14)),
                tickfont=dict(color='#1e293b', size=12)
            ),
            yaxis=dict(
                title=dict(text='Forecasted Value', font=dict(color='#1e293b', size=14)),
                tickfont=dict(color='#1e293b', size=12)
            )
        )
        st.plotly_chart(fig_fc, use_container_width=True)

# Tab 2: Geography
with tab2:
    st.markdown("### 📊 Geographic Distribution")
    st.info("🎯 **Objective**: Identify coverage gaps and resource allocation needs")
    
    if 'state' in df.columns:
        state_data = df.groupby('state')[metric_col].sum().reset_index().sort_values(metric_col, ascending=False)
        
        fig_geo = go.Figure(go.Bar(x=state_data['state'].head(15), y=state_data['total'].head(15) if 'total' in state_data else state_data[metric_col].head(15),
                                   marker=dict(color=state_data[metric_col].head(15), colorscale=[[0, '#2563eb'], [1, '#f97316']]),
                                   text=state_data[metric_col].head(15), texttemplate='%{text:,.0f}'))
        fig_geo.update_layout(
            title=dict(text="Top 15 States", font=dict(color='#1e293b', size=18)),
            template="plotly_white", 
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#1e293b', size=12),
            xaxis=dict(
                title=dict(text='State', font=dict(color='#1e293b', size=14)),
                tickfont=dict(color='#1e293b', size=12)
            ),
            yaxis=dict(
                title=dict(text='Total Count', font=dict(color='#1e293b', size=14)),
                tickfont=dict(color='#1e293b', size=12)
            )
        )
        st.plotly_chart(fig_geo, use_container_width=True)
    
    if dataset_type == 'enrolment' and 'age_0_5' in df.columns:
        age_df = pd.DataFrame({'age_group': ['0-5 Years', '5-17 Years', '18+ Years'],
                              'count': [df['age_0_5'].sum(), df['age_5_17'].sum(), df['age_18_greater'].sum()]})
        fig_age = px.pie(age_df, names='age_group', values='count', title="Age Distribution", hole=0.4,
                        color_discrete_sequence=['#2563eb', '#f97316', '#1e40af'])
        fig_age.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#1e293b', size=12),
            title=dict(font=dict(color='#1e293b', size=18))
        )
        st.plotly_chart(fig_age, use_container_width=True)

# Tab 3: Map
with tab3:
    st.markdown("### 🗺️ Interactive Heatmap")
    st.info("🎯 **Objective**: Visualize regional activity hotspots")
    
    if 'state' in df.columns:
        geojson = load_geojson("assets/india_states.geojson")
        if geojson:
            property_key = get_geojson_property_key(geojson)
            if property_key:
                df_map = map_state_names(df.copy())
                state_totals = df_map.groupby('state')[metric_col].sum().reset_index()
                
                try:
                    fig_map = px.choropleth(state_totals, geojson=geojson, locations='state',
                                           featureidkey=property_key, color=metric_col,
                                           color_continuous_scale=[[0, '#eff6ff'], [0.5, '#2563eb'], [1, '#f97316']], 
                                           hover_name='state')
                    fig_map.update_geos(fitbounds="locations", visible=False, bgcolor='white')
                    fig_map.update_layout(
                        height=700, 
                        paper_bgcolor='white', 
                        plot_bgcolor='white', 
                        geo=dict(bgcolor='white'),
                        font=dict(color='#1e293b', size=12),
                        title=dict(font=dict(color='#1e293b', size=18))
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.error(f"Map error: {e}")

# Tab 4: Insights
with tab4:
    st.markdown("### 💡 Actionable Insights")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #dbeafe, #bfdbfe); padding: 25px; border-radius: 15px; border: 2px solid #2563eb;'>
        <h3 style='color: #1e3a8a; margin-top: 0;'>🎯 Problem Alignment</h3>
        <p style='color: #1e293b;'><strong>Challenge:</strong> Unlock societal trends in Aadhaar enrolment and updates</p>
        <p style='color: #1e293b;'><strong>Solution:</strong> This dashboard provides:</p>
        <ul style='color: #1e293b;'>
            <li>✅ Real-time anomaly detection for operational issues</li>
            <li>✅ Predictive forecasting for resource planning</li>
            <li>✅ Geographic insights for targeted interventions</li>
            <li>✅ Demographic patterns for policy decisions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Key Findings")
    if 'date' in df.columns:
        ts_summary = df.groupby('date')[metric_col].sum()
        peak_val = ts_summary.max()
        peak_date = ts_summary.idxmax().strftime('%d %b %Y')
        st.success(f"**Peak Activity**: {int(peak_val):,} on {peak_date}")
    
    if 'state' in df.columns:
        top_state = df.groupby('state')[metric_col].sum().idxmax()
        st.success(f"**Top State**: {top_state}")

# Tab 5: Export
with tab5:
    st.markdown("### 📥 Data Export")
    st.info("Download processed data and insights for further analysis")
    
    @st.cache_data
    def convert_df(dataframe):
        return dataframe.to_csv(index=False).encode('utf-8')
    
    if st.button("📊 Export Current Dataset"):
        csv = convert_df(df)
        st.download_button("⬇️ Download CSV", csv, f"uidai_{dataset_type}_export.csv", "text/csv")
    
    if 'date' in df.columns:
        ts_export = df.groupby('date')[metric_col].sum().reset_index()
        if st.button("📈 Export Time Series"):
            ts_csv = convert_df(ts_export)
            st.download_button("⬇️ Download Time Series", ts_csv, "uidai_timeseries.csv", "text/csv")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'> UIDAI Data Hackathon 2026 </p>", unsafe_allow_html=True)
