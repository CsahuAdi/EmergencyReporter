import pandas as pd
import folium
from folium.plugins import HeatMap
from IPython.display import display

# Assuming df is the DataFrame with columns: 'Type', 'Latitude', 'Longitude'

# Step 1: Clean the data by removing rows with missing coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])

# Step 2: Identify the top 5 types based on frequency
top_types = df['Type'].value_counts().head(5).index

# Step 3: Define custom colors for each of the top 5 types
# Using hex colors for distinct visualization
color_map = {
    top_types[0]: '#FF0000',  # Red for the first type
    top_types[1]: '#00FF00',  # Green for the second type
    top_types[2]: '#0000FF',  # Blue for the third type
    top_types[3]: '#FFFF00',  # Yellow for the fourth type
    top_types[4]: '#FF00FF'   # Magenta for the fifth type
}

# Step 4: Calculate the center of the map using the mean latitude and longitude
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

# Step 5: Create a Folium map centered at the calculated location
seattle_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Step 6: For each of the top 5 types, create a heatmap layer with a specific color
for type in top_types:
    # Filter the DataFrame for the current type
    type_df = df[df['Type'] == type]
    
    # Prepare the data as a list of [latitude, longitude] pairs
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in type_df.iterrows()]
    
    # Define a gradient with a single dominant color for this type
    gradient = {
        '0.4': color_map[type],  # Start with the chosen color
        '1.0': color_map[type]   # Keep the same color for max intensity
    }
    
    # Add a heatmap layer for this type with the custom gradient
    HeatMap(
        heat_data,
        radius=8,              # Clean, smooth radius as in the GPT code
        blur=15,               # Default blur for smooth appearance
        max_zoom=18,           # Ensure clarity at different zoom levels
        gradient=gradient,     # Apply the custom color
        name=type              # Name the layer for LayerControl
    ).add_to(seattle_map)

# Step 7: Add LayerControl to allow toggling of each type's heatmap
folium.LayerControl().add_to(seattle_map)

# Step 8: Display the map inline using IPython display
display(seattle_map)
df.info()