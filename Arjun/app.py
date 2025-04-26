import streamlit as st
import folium
from streamlit_folium import st_folium
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import IsolationForest
import pydeck as pdk
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Seattle Emergency Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #dc3545;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #c82333;
    }
    .card {
        background-color: black;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .title {
        color: #2c3e50;
        font-weight: bold;
    }
    .subheader {
        color: #34495e;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 500;
        color: #34495e;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #dc3545;
        border-bottom: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    with open('emergency_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv('../cleaned_seattle.csv')
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'date', 'hour']].rename(columns={'date': 'day'}).assign(minute=0, second=0))
    return df

df = load_data()

# Main title
st.markdown("<h1 class='title'>🚨 Seattle Emergency Dashboard</h1>", unsafe_allow_html=True)

# Sidebar for controls and information
with st.sidebar:
    st.markdown("<h3 class='title'>Dashboard Controls</h3>", unsafe_allow_html=True)
    st.markdown("Monitor emergencies in Seattle with real-time predictions, heatmap analysis, and live trends.")
    st.markdown("### Instructions")
    st.markdown("- Prediction Map: Click on the map to predict emergency types")
    st.markdown("- Heatmap Analysis: Explore emergency density over time")
    st.markdown("- Live Monitoring: Monitor live emergency trends and anomalies")
    st.markdown("### Data Source")
    st.markdown("Seattle Emergency Services Data (CSV)")
    st.markdown("### About")
    st.markdown("Built with Streamlit, Folium, Pydeck, and Plotly")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📍 Prediction Map", "🌡 Heatmap Analysis", "📈 Live Monitoring"])

# Prediction Map Tab
with tab1:
    col_map, col_stats = st.columns([3, 1])
    with col_map:
        st.markdown("<div class='card'><h3 class='subheader'>📍 Interactive Emergency Prediction Map</h3>", unsafe_allow_html=True)
        seattle_map = folium.Map(location=[47.6062, -122.3321], zoom_start=12, tiles="CartoDB Positron")
        seattle_map.add_child(folium.LatLngPopup())
        map_data = st_folium(seattle_map, width=900, height=500, key="map")
        if map_data['last_clicked'] is not None:
            lat = map_data['last_clicked']['lat']
            lon = map_data['last_clicked']['lng']
            st.success(f"Location Selected: Latitude: {lat:.4f}, Longitude: {lon:.4f}")
            prediction = model.predict([[lat, lon]])[0]
            st.info(f"🚑 Predicted Emergency Type: {prediction}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_stats:
        st.markdown("<div class='card'><h3 class='subheader'>📊 Quick Stats</h3>", unsafe_allow_html=True)
        st.markdown("Total Emergencies: " + str(len(df)))
        st.markdown("Date Range: " + f"{df['datetime'].min().date()} to {df['datetime'].max().date()}")
        st.markdown("Anomaly Detection Model: Isolation Forest")
        st.markdown("</div>", unsafe_allow_html=True)

# Heatmap Analysis Tab
with tab2:
    st.markdown("<h3 class='subheader'>🌡 Emergency Calls Heatmap with Time Slider</h3>", unsafe_allow_html=True)
    st.subheader("Filter Options")
    call_types = df['Type'].unique()
    selected_types = st.multiselect("Select Emergency Call Types", call_types, default=call_types, key="types_multiselect")
    radius = st.slider("Heatmap Radius", 10, 100, 50, key="radius_slider")
    opacity = st.slider("Heatmap Opacity", 0.1, 1.0, 0.8, key="opacity_slider")
    color_schemes = {
        "Default": [[0, 0, 255], [255, 255, 0], [255, 0, 0]],
        "Cool": [[0, 255, 255], [0, 0, 255]],
        "Warm": [[255, 255, 0], [255, 0, 0]],
    }
    selected_scheme = st.selectbox("Select Color Scheme", list(color_schemes.keys()), key="color_scheme_select")
    start_time, end_time = st.slider(
        "Select Time Range",
        min_value=df['datetime'].min().to_pydatetime(),
        max_value=df['datetime'].max().to_pydatetime(),
        value=(df['datetime'].min().to_pydatetime(), df['datetime'].max().to_pydatetime()),
        format="DD MMM YY - HH:mm",
        key="time_slider"
    )
    # Filter data
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = df[['Latitude', 'Longitude', 'datetime', 'Type']]
    if 'last_range' not in st.session_state:
        st.session_state.last_range = None
    if 'last_types' not in st.session_state:
        st.session_state.last_types = None
    if (st.session_state.last_range != (start_time, end_time) or
        st.session_state.last_types != selected_types):
        filtered = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time) & (df['Type'].isin(selected_types))]
        filtered = filtered[['Latitude', 'Longitude', 'datetime', 'Type']]
        if len(filtered) > 50000:
            filtered = filtered.sample(50000)
        st.session_state.filtered_df = filtered
        st.session_state.last_range = (start_time, end_time)
        st.session_state.last_types = selected_types
    # Heatmap layer
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=st.session_state.filtered_df,
        get_position=["Longitude", "Latitude"],
        radiusPixels=radius,
        opacity=opacity,
        color_range=color_schemes[selected_scheme],
        pickable=True,
    )
    # Deck
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=pdk.ViewState(latitude=47.61, longitude=-122.33, zoom=10, pitch=50),
        layers=[heatmap_layer],
        tooltip={"text": "Latitude: {Latitude}\nLongitude: {Longitude}\nType: {Type}"}
    )
    st.pydeck_chart(deck)
    # Line chart
    counts = st.session_state.filtered_df.groupby(st.session_state.filtered_df['datetime'].dt.date).size().reset_index(name='count')
    fig = px.line(counts, x='datetime', y='count', title="Emergency Calls Over Time")
    st.plotly_chart(fig)

# Live Monitoring Tab
with tab3:
    st.markdown("<div class='card'><h3 class='subheader'>📈 Live Emergency Trends & Anomaly Detection</h3>", unsafe_allow_html=True)
    def live_plot():
        hourly_emergencies = df.set_index('datetime').resample('H').size()
        X = hourly_emergencies.values.reshape(-1, 1)
        clf = IsolationForest(contamination=0.001, random_state=42)
        clf.fit(X)
        anomalies = clf.predict(X)
        anomalous_dates = hourly_emergencies.index[anomalies == -1]
        anomalous_counts = hourly_emergencies[anomalies == -1]
        plot_placeholder = st.empty()
        for i in range(24, len(hourly_emergencies)):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(hourly_emergencies.index[:i], hourly_emergencies.values[:i],
                    label='Hourly Emergencies', color='#2c3e50')
            ax.scatter(anomalous_dates[anomalous_dates <= hourly_emergencies.index[i]],
                       anomalous_counts[anomalous_dates <= hourly_emergencies.index[i]],
                       color='#dc3545', label='Anomalies', s=100)
            ax.set_title('Hourly Emergency Trends with Anomaly Detection', fontsize=14, pad=15)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Number of Emergencies', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            plot_placeholder.pyplot(fig)
            if hourly_emergencies.index[i] in anomalous_dates:
                st.error(f"🚨 RED ALERT! Anomaly detected at {hourly_emergencies.index[i]}")
            time.sleep(0.05)
    if st.button('Start Live Monitoring 🚨', key="monitor_button"):
        live_plot()
    st.markdown("</div>", unsafe_allow_html=True)