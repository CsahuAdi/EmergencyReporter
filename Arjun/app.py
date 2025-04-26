import pandas as pd
import folium
from folium.plugins import HeatMap

from IPython.display import display

# Load dataset
df = pd.read_csv('seattle.csv')  # Uncomment if using a CSV file

# Assuming df is already loaded
df = df.dropna(subset=['Latitude', 'Longitude'])

# Optional: Filter only Seattle area coordinates
# df = df[(df['Latitude'] >= 47.4) & (df['Latitude'] <= 47.8) &
#         (df['Longitude'] >= -122.5) & (df['Longitude'] <= -122.2)]

# Create map centered at Seattle
seattle_map = folium.Map(location=[47.6062, -122.3321], zoom_start=12)

# Prepare heatmap data
heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]

# Add heatmap layer
HeatMap(heat_data, radius=8).add_to(seattle_map)

# Display map inline
seattle_map
