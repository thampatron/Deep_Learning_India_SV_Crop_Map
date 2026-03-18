import geopandas as gpd
import pandas as pd
import random
import ee
import geemap
from shapely.geometry import Point 

# Initialize Earth Engine
ee.Initialize()

# Set random seed for reproducibility
random.seed(42)

def generate_random_points_in_india(shapefile_path, num_points=20000):
    # Load the India shapefile
    india_shapefile = gpd.read_file(shapefile_path)
    
    # Convert to a bounding box for efficient sampling
    india_bounds = india_shapefile.total_bounds  # (minx, miny, maxx, maxy)
    
    points = []
    while len(points) < num_points:
        # Generate a random point within the bounding box
        lat = random.uniform(india_bounds[1], india_bounds[3])
        lon = random.uniform(india_bounds[0], india_bounds[2])
        
        # Create a shapely Point geometry
        point = Point(lon, lat)
        
        # Check if the point is within India using the shapefile
        if point.within(india_shapefile.unary_union):
            points.append((lat, lon))

    return pd.DataFrame(points, columns=['latitude', 'longitude'])

def filter_points_with_cropland(points_df, cropland_layer):
    # Convert points to Earth Engine objects
    point_features = [ee.Geometry.Point([lon, lat]) for lat, lon in zip(points_df['latitude'], points_df['longitude'])]
    points_ee = ee.FeatureCollection(point_features)

    # Apply cropland filter using the ESA WorldCover layer
    cropland = ee.Image(cropland_layer)
    
    # ESA WorldCover classifies cropland as value 40 (cropland class code)
    cropland_mask = cropland.eq(40)  
    
    # Create a masked layer for points
    cropland_points = points_ee.filter(ee.Filter.eq(cropland_mask.reduceRegion(
        reducer=ee.Reducer.first(), 
        geometry=ee.Geometry.MultiPoint(point_features), 
        scale=10  # 10 meter resolution for ESA WorldCover
    ).values()))

    # Extract the valid points
    valid_points = cropland_points.getInfo()['features']
    
    filtered_points = [(p['geometry']['coordinates'][1], p['geometry']['coordinates'][0]) for p in valid_points]
    return pd.DataFrame(filtered_points, columns=['latitude', 'longitude'])

def main(shapefile_path, cropland_layer, output_csv):
    # Generate random points
    points_df = generate_random_points_in_india(shapefile_path)
    
    # Filter points that fall within cropland
    filtered_points_df = filter_points_with_cropland(points_df, cropland_layer)
    
    # Save the filtered points to CSV
    filtered_points_df.to_csv(output_csv, index=False)
    print(f"Saved {len(filtered_points_df)} points to {output_csv}")

# Example usage


shapefile_path = 'SecondLevelIndiaShp/secondLevelIndiaShp.shp'
cropland_layer = 'ESA/WorldCover/v100/2020'  # ESA WorldCover 2020 dataset
output_csv = 'filtered_cropland_points_esa.csv'

ee.Authenticate()
ee.Initialize(project='co2-sensing')

main(shapefile_path, cropland_layer, output_csv)
