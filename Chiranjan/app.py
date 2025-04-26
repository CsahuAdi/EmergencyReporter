import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import tempfile
import os

st.set_page_config(page_title="Emergency Hotspot Predictor", layout="wide")

st.title("🚑 Emergency Call Hotspot Prediction App")

# Upload raw emergency CSV
uploaded_file = st.file_uploader("Upload Raw Emergency Data (CSV)", type=['csv'])

if uploaded_file:
    st.success("✅ Emergency call data uploaded!")
    
    df_raw = pd.read_csv(uploaded_file)
    st.write("First few rows of raw data:")
    st.dataframe(df_raw.head())

    # Preprocessing
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    if 'latitude' not in df_raw.columns or 'longitude' not in df_raw.columns:
        st.error("❌ The uploaded CSV must contain 'latitude' and 'longitude' columns.")
        st.stop()

    df_raw['lat_bin'] = df_raw['latitude'].round(2)
    df_raw['lon_bin'] = df_raw['longitude'].round(2)

    if 'datetime' not in df_raw.columns:
        st.error("❌ The uploaded CSV must contain 'Datetime' column (case-insensitive).")
        st.stop()

    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_raw['year'] = df_raw['datetime'].dt.year
    df_raw['month'] = df_raw['datetime'].dt.month
    df_raw['date'] = df_raw['datetime'].dt.day
    df_raw['hour'] = df_raw['datetime'].dt.hour

    # Create Demand Data
    demand = df_raw.groupby(['year', 'month', 'date', 'hour', 'lat_bin', 'lon_bin']).size().reset_index(name='num_calls')

    # Prepare data for LSTM
    feature_cols = ['hour', 'lat_bin', 'lon_bin']
    target_col = 'num_calls'

    X = demand[feature_cols].values
    y = demand[target_col].values

    # Create sequences
    X_seq = []
    y_seq = []

    sequence_length = 24

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Validation data shape: {X_val.shape}")

    # Train LSTM model
    with st.spinner("Training LSTM model..."):
        model = Sequential()
        model.add(LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64, callbacks=[early_stop], verbose=0)

    st.success("✅ Model training complete!")

    # Save trained model temporarily
    temp_model_path = os.path.join(tempfile.gettempdir(), 'temp_model.h5')
    model.save(temp_model_path)

    # Upload future_df CSV
    st.subheader("📅 Upload Future Data (for prediction)")

    future_file = st.file_uploader("Upload Future Data CSV", type=['csv'])

    if future_file:
        future_df = pd.read_csv(future_file)
        st.write("First few rows of future data:")
        st.dataframe(future_df.head())

        # Check required columns
        required_cols = ['hour', 'lat_bin', 'lon_bin']
        if not all(col in future_df.columns for col in required_cols):
            st.error(f"❌ The uploaded future_df must contain columns: {required_cols}")
            st.stop()

        # Prepare input
        X_future = future_df[required_cols].values
        X_future = X_future.reshape((X_future.shape[0], 1, X_future.shape[1]))

        # Load model and predict
        trained_model = load_model(temp_model_path)
        y_future_pred = trained_model.predict(X_future, verbose=1)

        future_df['predicted_num_calls'] = y_future_pred.flatten()

        st.success("✅ Prediction complete!")
        st.write("Sample Predictions:")
        st.dataframe(future_df[['datetime', 'lat_bin', 'lon_bin', 'predicted_num_calls']].head())

        # Create Heatmap
        st.subheader("🌎 Emergency Hotspot Heatmap")

        seattle_map = folium.Map(location=[47.6062, -122.3321], zoom_start=11)

        heat_data = future_df[['lat_bin', 'lon_bin', 'predicted_num_calls']].values.tolist()

        HeatMap(heat_data, radius=8, blur=15, max_zoom=10).add_to(seattle_map)

        # Show map
        from streamlit_folium import st_folium
        st_folium(seattle_map, width=800, height=600)

        # Download updated future_df
        csv = future_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predicted Future Data",
            data=csv,
            file_name='future_emergency_predictions.csv',
            mime='text/csv'
        )
