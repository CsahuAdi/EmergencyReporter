import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta
import plotly.express as px
import time

# Title
st.title("Seattle Emergency Calls Heatmap with Time Slider")

# Load and preprocess data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'date', 'hour']].rename(columns={'date': 'day'}).assign(minute=0, second=0))
    # Derive priority based on top Type occurrences (not used for scatterplot but kept for potential future use)
    type_counts = df['Type'].value_counts()
    top_types = type_counts.head(2).index  # Top 2 most frequent Types
    df['priority'] = df['Type'].apply(lambda x: 'High' if x in top_types else 'Low')
    return df

df = load_data('cleaned_seattle.csv')

# Initialize session state
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = df[['Latitude', 'Longitude', 'datetime', 'Type', 'priority']]
if 'last_range' not in st.session_state:
    st.session_state.last_range = None
if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False

# Sidebar controls
st.sidebar.header("Filter Options")

# Feature 1: Type Filter for Emergency Call Categories
call_types = df['Type'].unique()
selected_types = st.sidebar.multiselect("Select Emergency Call Types", call_types, default=call_types)

# Feature 2: Dynamic Radius and Opacity Controls
radius = st.sidebar.slider("Heatmap Radius", 10, 100, 50)
opacity = st.sidebar.slider("Heatmap Opacity", 0.1, 1.0, 0.8)

# Feature 6: Custom Color Schemes
color_schemes = {
    "Default": [[0, 0, 255], [255, 255, 0], [255, 0, 0]],  # Blue to Yellow to Red
    "Cool": [[0, 255, 255], [0, 0, 255]],  # Cyan to Blue
    "Warm": [[255, 255, 0], [255, 0, 0]],  # Yellow to Red
}
selected_scheme = st.sidebar.selectbox("Select Color Scheme", list(color_schemes.keys()))

# Time slider
start_time, end_time = st.sidebar.slider(
    "Select Time Range",
    min_value=df['datetime'].min().to_pydatetime(),
    max_value=df['datetime'].max().to_pydatetime(),
    value=(df['datetime'].min().to_pydatetime(), df['datetime'].max().to_pydatetime()),
    format="DD MMM YY - HH:mm"
)

# Update filtered data if slider or filters change
if (st.session_state.get('last_range') != (start_time, end_time) or
    st.session_state.get('last_types') != selected_types):
    filtered = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time) & (df['Type'].isin(selected_types))]
    filtered = filtered[['Latitude', 'Longitude', 'datetime', 'Type', 'priority']]
    if len(filtered) > 50000:
        filtered = filtered.sample(50000)
    st.session_state.filtered_df = filtered
    st.session_state.last_range = (start_time, end_time)
    st.session_state.last_types = selected_types

# # Feature 3: Time Animation
# if st.sidebar.button("Play Animation"):
#     st.session_state.animation_running = True

# if st.session_state.animation_running:
#     current_time = start_time
#     step = timedelta(hours=1)  # Hourly steps
#     while current_time <= end_time and st.session_state.animation_running:
#         filtered = df[(df['datetime'] >= current_time) & (df['datetime'] < current_time + step) & (df['Type'].isin(selected_types))]
#         filtered = filtered[['Latitude', 'Longitude', 'datetime', 'Type', 'priority']]
#         if len(filtered) > 50000:
#             filtered = filtered.sample(50000)
#         st.session_state.filtered_df = filtered
#         st.rerun()  # Updated from st.experimental_rerun()
#         current_time += step
#         time.sleep(0.5)  # Animation speed
#     st.session_state.animation_running = False

# # Stop animation button
# if st.session_state.animation_running:
#     if st.sidebar.button("Stop Animation"):
#         st.session_state.animation_running = False
#         st.rerun()  # Updated from st.experimental_rerun()

# Feature 5: Heatmap with Interactive Tooltip
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=st.session_state.filtered_df,
    get_position=["Longitude", "Latitude"],
    radiusPixels=radius,
    opacity=opacity,
    color_range=color_schemes[selected_scheme],
    pickable=True,
)

# Create Deck
deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=pdk.ViewState(latitude=47.61, longitude=-122.33, zoom=10, pitch=50),
    layers=[heatmap_layer],
    tooltip={"text": "Latitude: {Latitude}\nLongitude: {Longitude}\nType: {Type}"}
)

st.pydeck_chart(deck)

# Feature 9: Time-Based Aggregation (Daily Trends)
filtered = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time) & (df['Type'].isin(selected_types))]
counts = filtered.groupby(filtered['datetime'].dt.date).size().reset_index(name='count')
fig = px.line(counts, x='datetime', y='count', title="Emergency Calls Over Time")
st.plotly_chart(fig)

# Footer
st.markdown("---")
st.info("Move the slider to see how the density of emergency calls changes over time. Use the filters to customize the visualization.")