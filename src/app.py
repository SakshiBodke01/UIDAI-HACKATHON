import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO

from preprocess import load_dataset
from utils import inject_css, kpi_row, cache_df
from analytics import zscore_anomalies, moving_average, arima_forecast
from geo import load_geojson, map_state_names, get_geojson_property_key


# ===================== GOOGLE DRIVE DATA LINKS =====================
DATA_URLS = {
    "enrolment": "https://drive.google.com/uc?export=download&id=1VXQ61eDwrmeVML31g_ubPU__LSPE6FWu",
    "demographic": "https://drive.google.com/uc?export=download&id=1FCoAmEtzposGd4VgNyeI_I_jwZWhwzGe",
    "biometric": "https://drive.google.com/uc?export=download&id=1cdQM1TGvlg_ed1PhahwaxQbnHeHW1tbm",
}
# ===================================================================


st.set_page_config(
    page_title="UIDAI Insights Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== SIDEBAR =====================
st.sidebar.markdown("## ⚙️ Dashboard Controls")
st.sidebar.markdown("---")

dataset_type = st.sidebar.selectbox(
    "📁 Dataset Type",
    ["enrolment", "biometric", "demographic"]
)

st.sidebar.markdown("📂 **Data Source:** Google Drive (Cloud)")
st.sidebar.markdown("---")

anomaly_threshold = st.sidebar.slider("🚨 Anomaly Sensitivity", 1.5, 4.0, 2.5, 0.1)
ma_window = st.sidebar.slider("📉 Smoothing Window", 3, 30, 7, 1)
forecast_steps = st.sidebar.slider("🔮 Forecast Horizon", 3, 30, 7, 1)

# ===================== LOAD DATA =====================
try:
    file_path = DATA_URLS[dataset_type]
    df = load_dataset(file_path, dataset_type)
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

df = cache_df(df)

metric_col = {
    'enrolment': 'total_enrolment',
    'biometric': 'total_biometric',
    'demographic': 'total_demographic'
}[dataset_type]

# ===================== KPIs =====================
st.markdown("## 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📝 Total Records", f"{len(df):,}")

with col2:
    st.metric(
        f"🎯 {metric_col.replace('_', ' ').title()}",
        f"{int(df[metric_col].sum()):,}",
        f"{int(df[metric_col].mean()):,} avg"
    )

with col3:
    st.metric("🗺️ States", df['state'].nunique() if 'state' in df.columns else 0)

with col4:
    st.metric("📍 Districts", df['district'].nunique() if 'district' in df.columns else 0)

st.markdown("---")

# ===================== TABS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Trends", "📊 Geography", "🗺️ Map", "💡 Insights", "📥 Export"]
)

# ===================== TAB 1: TRENDS =====================
with tab1:
    if 'date' in df.columns:
        ts = df.groupby('date')[metric_col].sum().reset_index().sort_values('date')
        ts['ma'] = moving_average(ts[metric_col], window=ma_window)

        anomalies_idx, _ = zscore_anomalies(ts[metric_col], threshold=anomaly_threshold)
        ts['is_anomaly'] = False
        ts.loc[ts.index[anomalies_idx], 'is_anomaly'] = True

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts['date'], y=ts[metric_col], mode='lines', name='Daily'))
        fig.add_trace(go.Scatter(x=ts['date'], y=ts['ma'], mode='lines', name='Moving Avg'))
        fig.add_trace(go.Scatter(
            x=ts[ts['is_anomaly']]['date'],
            y=ts[ts['is_anomaly']][metric_col],
            mode='markers',
            name='Anomalies'
        ))
        st.plotly_chart(fig, use_container_width=True)

        fc = arima_forecast(
            ts[['date', metric_col]].rename(columns={metric_col: 'value'}),
            'value',
            steps=forecast_steps
        )
        st.line_chart(fc.set_index('date')['forecast'])

# ===================== TAB 2: GEOGRAPHY =====================
with tab2:
    if 'state' in df.columns:
        state_data = df.groupby('state')[metric_col].sum().reset_index()
        fig = px.bar(state_data.sort_values(metric_col, ascending=False).head(15),
                     x='state', y=metric_col)
        st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 3: MAP =====================
with tab3:
    geojson = load_geojson("assets/india_states.geojson")
    if geojson and 'state' in df.columns:
        df_map = map_state_names(df)
        totals = df_map.groupby('state')[metric_col].sum().reset_index()

        fig = px.choropleth(
            totals,
            geojson=geojson,
            locations='state',
            featureidkey=get_geojson_property_key(geojson),
            color=metric_col
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 4: INSIGHTS =====================
with tab4:
    if 'date' in df.columns:
        peak = df.groupby('date')[metric_col].sum().idxmax()
        st.success(f"📈 Peak Activity Date: {peak.strftime('%d %b %Y')}")

# ===================== TAB 5: EXPORT =====================
with tab5:
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)
    st.download_button(
        "⬇️ Download Dataset",
        csv,
        f"uidai_{dataset_type}.csv",
        "text/csv"
    )

st.markdown("---")
st.markdown("<p style='text-align:center;'>UIDAI Data Hackathon 2026</p>",
            unsafe_allow_html=True)
