import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from streamlit_folium import st_folium

# Title
st.title("Emergency Calls Prediction and Heatmap Visualization")

# Upload CSV
uploaded_file = st.file_uploader("Upload your final ready data CSV", type=["csv"])

if uploaded_file is not None:
    # ---------- 1 .  Load & echo data ----------
    data = pd.read_csv(uploaded_file)
    st.subheader("First few rows of uploaded data")
    st.dataframe(data.head())

    # ---------- 2 .  Feature / target split ----------
    # Anything except the target column is a feature
    target_col = "target_num_calls"
    feature_cols = [c for c in data.columns if c != target_col]

    X = data[feature_cols].values.astype("float32")
    y = data[target_col].values.astype("float32")

    # ---------- 3 .  Train /-validation split ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- 4 .  Simple dense model  ----------
    # (There is no sequence axis any more, so Dense works fine;
    #  swap back to an LSTM if you REALLY want one.)
    best_params = {
        "n_units": 128,
        "dropout_rate": 0.20,
        "learning_rate": 1e-4,
        "batch_size": 128,
    }
    final_model = Sequential(
        [
            Dense(best_params["n_units"], activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(best_params["dropout_rate"]),
            Dense(1),
        ]
    )
    final_model.compile(
        optimizer=Adam(learning_rate=best_params["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )

    with st.spinner("Training the model …"):
        final_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=best_params["batch_size"],
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0,
        )

    # ---------- 5 .  Make predictions ----------
    y_pred = final_model.predict(X, verbose=0).flatten()
    future_df = data.copy()
    future_df["predicted_num_calls"] = y_pred

    st.subheader("First few rows of future_df")
    st.dataframe(future_df.head())

    # ---------- 6 .  Build heat-map ----------
    st.subheader("Predicted Emergency-Calls Heat-map")

    # Use the most-recent location bins (t-1) as map coordinates
    lat_col = "lat_bin_t-1"
    lon_col = "lon_bin_t-1"

    seattle_map = folium.Map(location=[47.6062, -122.3321], zoom_start=11)
    heat_data = future_df[[lat_col, lon_col, "predicted_num_calls"]].values.tolist()
    HeatMap(heat_data, radius=8, blur=15, max_zoom=10).add_to(seattle_map)

    st_folium(seattle_map, width=700, height=500)

    # ---------- 7 .  Download button ----------
    csv = future_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download future_df as CSV",
        data=csv,
        file_name="future_predictions.csv",
        mime="text/csv",
    )
