import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import DateTime as dt

import folium
from folium.plugins import HeatMap

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Make sure your data is ready
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

best_params = {'n_units': 64, 'dropout_rate': 0.1904136995405557, 'learning_rate': 7.506755072870304e-05, 'batch_size': 128}

final_model = Sequential()
final_model.add(LSTM(best_params["n_units"], input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
final_model.add(Dropout(best_params["dropout_rate"]))
final_model.add(Dense(1))
final_model.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]), loss='mse', metrics=['mae'])

final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=best_params["batch_size"],
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

y_val_pred = final_model.predict(X_val)
y_val_pred_rounded = np.round(y_val_pred).flatten()
y_val_true_rounded = np.round(y_val).flatten()

import numpy as np
import pandas as pd

# Define Seattle bounding box
lat_min, lat_max = 47.4919, 47.7341
lon_min, lon_max = -122.4594, -122.2244

# Step size for grid (smaller step = more fine grid)
lat_steps = np.arange(lat_min, lat_max, 0.15)
lon_steps = np.arange(lon_min, lon_max, 0.15)

# Create mesh grid
grid_lats, grid_lons = np.meshgrid(lat_steps, lon_steps)

# Flatten into list of points
grid_points = np.vstack([grid_lats.ravel(), grid_lons.ravel()]).T

print(f"Total Grid Points: {len(grid_points)}")

future_dates = pd.date_range(start='2025-04-26', end='2030-04-26', freq='H')

# Create future DataFrame
future_df = pd.DataFrame({'datetime': np.repeat(future_dates, len(grid_points))})

# Expand lat/lon for each timestamp
future_df['lat_bin'] = np.tile(grid_points[:, 0], len(future_dates))
future_df['lon_bin'] = np.tile(grid_points[:, 1], len(future_dates))

# Add date/time features
future_df['year'] = future_df['datetime'].dt.year
future_df['month'] = future_df['datetime'].dt.month
future_df['date'] = future_df['datetime'].dt.day
future_df['hour'] = future_df['datetime'].dt.hour

print(future_df.shape)
future_df.head()

X_future = future_df[['hour', 'lat_bin', 'lon_bin']].values

# If your LSTM needs 24 hours sequence, you need to prepare sliding windows (can simulate simple predict per hour for now)

# For first version, reshape to (samples, 1, features) for basic LSTM prediction
X_future = X_future.reshape((X_future.shape[0], 1, X_future.shape[1]))

# Use your trained LSTM model
y_future_pred = final_model.predict(X_future, verbose=1)

# Attach predictions
future_df['predicted_num_calls'] = y_future_pred.flatten()

future_df.to_csv('future_emergency_grid_predictions.csv', index=False)

import folium
from folium.plugins import HeatMap

# Base Map
seattle_map = folium.Map(location=[47.6062, -122.3321], zoom_start=11)

# Prepare heatmap data (lat, lon, intensity)
heat_data = future_df[['lat_bin', 'lon_bin', 'predicted_num_calls']].values.tolist()

# Add heatmap layer
HeatMap(heat_data, radius=8, blur=15, max_zoom=10).add_to(seattle_map)

# Save HTML file
seattle_map.save('seattle_future_emergency_heatmap.html')

# Display in notebook
seattle_map