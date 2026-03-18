import pandas as pd
import os
import math
import re
from pathlib import Path
import datetime
from sklearn.neighbors import BallTree
import numpy as np

def extract_head_angle(image_path):
    """Extract heading angle from image path using regex."""
    match = re.search(r'head([-\d.]+)', image_path)
    if match:
        return float(match.group(1))
    return None

def get_new_coordinates(lat, lon, heading, distance):
    """
    Calculate new coordinates given a starting point, heading and distance.
    
    Args:
        lat, lon: Starting coordinates in degrees
        heading: Heading in degrees (0 is North, 90 is East)
        distance: Distance to travel in meters
    
    Returns:
        (new_lat, new_lon) in degrees
    """
    # Convert to radians
    lat_rad = math.radians(float(lat))
    lon_rad = math.radians(float(lon))
    heading_rad = math.radians(float(heading))
    
    # Earth's radius in meters
    R = 6371000
    
    # Angular distance
    d = distance / R
    
    # Calculate new latitude
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(d) +
        math.cos(lat_rad) * math.sin(d) * math.cos(heading_rad)
    )
    
    # Calculate new longitude
    new_lon_rad = lon_rad + math.atan2(
        math.sin(heading_rad) * math.sin(d) * math.cos(lat_rad),
        math.cos(d) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )
    
    # Convert back to degrees
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)
    
    return new_lat, new_lon

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in meters."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Earth's radius in meters
    return c * r

def filter_by_proximity(df, distance_threshold=100):
    """
    Filter points that are within the given distance threshold of another point.
    Uses BallTree for efficient nearest neighbor search.
    
    Returns a dataframe with filtered points.
    """
    if len(df) == 0:
        print("No points to filter by proximity")
        return df
        
    print(f"Filtering points within {distance_threshold}m of each other...")
    
    try:
        # Try to use the 20m projected coordinates if they exist
        if 'latitude_20' in df.columns and 'longitude_20' in df.columns:
            print("Using projected 20m coordinates for proximity filtering")
            coords = df[['latitude_20', 'longitude_20']].values
        else:
            print("Using original coordinates for proximity filtering")
            coords = df[['latitude', 'longitude']].values
            
        # Convert lat/lon to radians for BallTree
        coords_rad = np.radians(coords)
        
        # Create a BallTree
        tree = BallTree(coords_rad, metric='haversine')
        
        # Find all points within distance_threshold meters
        # Convert distance to radians for the query
        distance_rad = distance_threshold / 6371000.0
        indices = tree.query_radius(coords_rad, distance_rad)
        
        # Keep track of which points to keep
        to_keep = np.ones(len(df), dtype=bool)
        
        # Process each point and its neighbors
        for i, neighbors in enumerate(indices):
            # Skip if this point has already been marked for removal
            if not to_keep[i]:
                continue
            
            # Mark neighbors for removal (excluding self)
            for neighbor in neighbors:
                if neighbor != i:  # Don't remove self
                    to_keep[neighbor] = False
        
        # Apply the filter
        filtered_df = df.loc[to_keep].copy()
        
        print(f"Removed {len(df) - len(filtered_df)} points, kept {len(filtered_df)} points")
        return filtered_df
    except Exception as e:
        print(f"Error in proximity filtering: {e}")
        print("Returning original dataset without proximity filtering")
        return df

def parse_date(date_str):
    """Parse date string in the format YYYY-MM."""
    try:
        # Try standard format YYYY-MM
        return datetime.datetime.strptime(date_str, "%Y-%m")
    except ValueError:
        try:
            # Try alternative format with day YYYY-MM-DD 
            return datetime.datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
        except (ValueError, IndexError):
            return None

def is_in_kharif_2023(date_str):
    """Check if the date is within Kharif season 2023 (June-November)."""
    if date_str is None or pd.isna(date_str):
        return False
        
    date = parse_date(str(date_str))
    if date is None:
        return False
    
    return (date.year == 2023 and 6 <= date.month <= 11)

def process_csv(input_path):
    """Process a single CSV file and add new columns."""
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Initial rows: {len(df)}")
    
    # First, filter for only "growing" in fallow_label column
    if 'fallow_label' in df.columns:
        non_growing_count = len(df) - len(df[df['fallow_label'] == 'growing'])
        # df = df[df['fallow_label'] == 'growing']
        print(f"After filtering for 'growing' label: {len(df)} (removed {non_growing_count} non-growing points)")
    else:
        print("Warning: 'fallow_label' column not found. Unable to filter by growing status.")
    
    # Then filter by date - Kharif 2023 (June-November) if 'date' column exists
    if len(df) > 0:
        if 'date' in df.columns:
            before_count = len(df)
            # df = df[df['date'].apply(is_in_kharif_2023)]
            print(f"After date filtering (2023 June-November): {len(df)} (removed {before_count - len(df)} points)")
        else:
            print("Warning: 'date' column not found in CSV. Skipping date filtering.")
            
            # Try to extract date from image path as fallback
            try:
                # Extract date pattern like 'date2023-06' from image path
                df['extracted_date'] = df['image_path'].str.extract(r'date(\d{4}-\d{2})')
                print(f"Extracted dates from image paths")
                
                # Filter using extracted dates
                if 'extracted_date' in df.columns:
                    before_count = len(df)
                    # df = df[df['extracted_date'].apply(is_in_kharif_2023)]
                    print(f"After date filtering using extracted dates: {len(df)} (removed {before_count - len(df)} points)")
            except Exception as e:
                print(f"Error extracting dates from image paths: {e}")
                print("Proceeding without date filtering")
    
    if len(df) == 0:
        print("No data left after filtering")
        return df
    
    # Extract heading angles
    df['head'] = df['image_path'].apply(extract_head_angle)
    
    # Calculate new coordinates for different distances
    distances = [15, 20, 30]  # meters
    
    for dist in distances:
        # Use list comprehension with proper data access
        new_coords = [
            get_new_coordinates(row['latitude'], row['longitude'], row['head'], dist)
            for _, row in df.iterrows()
        ]
        
        df[f'latitude_{dist}'] = [coord[0] for coord in new_coords]
        df[f'longitude_{dist}'] = [coord[1] for coord in new_coords]
    
    return df

def main():
    # Configure input and output folders
    # NOTE I REMOVED DATE FILTERING FOR THIS PASS
    INPUT_FOLDER = "/home/laguarta_jordi/sean7391/streetview_highres/kharif_rabi_2023_train"  # User should modify this
    OUTPUT_FOLDER = "/home/laguarta_jordi/sean7391/streetview_highres/kharif_rabi_2023_train_20m"  # User should modify this
    input_dir = Path(INPUT_FOLDER)
    output_dir = Path(OUTPUT_FOLDER)
    
    # Verify input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_FOLDER}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List to store all dataframes
    all_dfs = []
    processed_count = 0
    error_count = 0
    
    # Process each CSV file
    csv_files = list(input_dir.glob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {INPUT_FOLDER}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        
        try:
            # Process the CSV
            df = process_csv(csv_file)
            
            # Skip empty dataframes
            if len(df) == 0:
                print(f"No data to process in {csv_file} after filtering")
                continue
            
            # Extract heading angles and calculate coordinates if not done in process_csv
            if 'head' not in df.columns:
                df['head'] = df['image_path'].apply(extract_head_angle)
                
                # Calculate new coordinates for different distances
                distances = [15, 20, 30]  # meters
                for dist in distances:
                    new_coords = [
                        get_new_coordinates(row['latitude'], row['longitude'], row['head'], dist)
                        for _, row in df.iterrows()
                    ]
                    
                    df[f'latitude_{dist}'] = [coord[0] for coord in new_coords]
                    df[f'longitude_{dist}'] = [coord[1] for coord in new_coords]
            
            # Save individual processed CSV
            output_path = output_dir / csv_file.name
            df.to_csv(output_path, index=False)
            print(f"Saved processed file to {output_path}")
            
            # Add to list of all dataframes
            all_dfs.append(df)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            error_count += 1
            continue
    
    print(f"\nSummary: Successfully processed {processed_count} files, encountered errors in {error_count} files")
    
    # Combine all dataframes
    if all_dfs:
        print("\nCombining all dataframes...")
        total_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total combined rows: {len(total_df)}")
        
        if len(total_df) > 0:
            # Filter by proximity (100m)
            filtered_df = filter_by_proximity(total_df, distance_threshold=50)
            
            # Save the combined filtered file
            total_path = output_dir / 'total.csv'
            filtered_df.to_csv(total_path, index=False)
            print(f"Saved combined and filtered file to {total_path} with {len(filtered_df)} rows")
        else:
            print("No data to save after combining dataframes")
    else:
        print("No data found to process after filtering")

if __name__ == "__main__":
    main()