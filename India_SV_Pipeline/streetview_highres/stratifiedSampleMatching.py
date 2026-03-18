import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape
import json

from scipy.spatial import cKDTree  # KDTree for nearest-neighbor lookups
import random

def remove_existing_matches(df_inferred, existing_batches):
    """
    Remove points that exist in any of the previous batch CSVs.
    
    Args:
        df_inferred: DataFrame with inferred points
        existing_batches: List of CSV filenames containing previously matched points
    
    Returns:
        DataFrame with points from existing batches removed
    """
    if not existing_batches:
        return df_inferred
    
    # Create a set of (lat, lon) tuples for fast lookup
    existing_points = set()
    for batch_file in existing_batches:
        if os.path.exists(batch_file):
            batch_df = pd.read_csv(batch_file)
            # Add both original points and matched random points
            points = zip(batch_df['latitude'], batch_df['longitude'])
            existing_points.update(points)
            if 'matched_random_lat' in batch_df.columns:
                random_points = zip(batch_df['matched_random_lat'], 
                                  batch_df['matched_random_lon'])
                existing_points.update(random_points)
    
    # Create mask for points not in existing_points
    mask = ~df_inferred.apply(
        lambda row: (row['latitude'], row['longitude']) in existing_points, 
        axis=1
    )
    
    return df_inferred[mask]

def check_min_distance(point_coords, existing_coords, min_distance_meters=50):
    """
    Check if a point is at least min_distance_meters away from all existing points.
    
    Args:
        point_coords: tuple of (lon, lat)
        existing_coords: array of shape (N, 2) with existing [lon, lat] coordinates
        min_distance_meters: minimum distance in meters
    
    Returns:
        bool: True if point meets minimum distance requirement
    """
    if len(existing_coords) == 0:
        return True
        
    # Convert lat/lon differences to approximate meters
    # Using rough conversion: 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
    lat = point_coords[1]
    lon_scale = np.cos(np.radians(lat)) * 111000
    lat_scale = 111000
    
    # Scale coordinate differences
    diff = existing_coords - point_coords
    diff[:, 0] *= lon_scale  # Scale longitude differences
    diff[:, 1] *= lat_scale  # Scale latitude differences
    
    # Calculate distances in meters
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    return np.all(distances >= min_distance_meters)

def load_inferred_fallow(inferred_folder):
    """
    Loads all .csv files from inferred_folder, filters to 'growing' (if you want),
    extracts lat/lon from 'image_path', parses the date as year/month, 
    and assigns season_inferred plus season_year. 
    Returns a DataFrame with columns:
      ['latitude', 'longitude', 'year_inferred', 'month_inferred', 
       'season_inferred', 'season_year', ...other columns...].
    """
    
    # Regex for lat/lon in 'image_path'
    lat_pattern  = re.compile(r'GSVLat(-?\d+\.\d+)')
    lon_pattern  = re.compile(r'GSVLon(-?\d+\.\d+)')
    # Regex for date in 'image_path' (e.g. "date2022-08")
    date_pattern = re.compile(r'date(\d{4}-\d{2})')
    
    # 1) Read all CSVs
    csv_files = glob.glob(os.path.join(inferred_folder, "*.csv"))
    df_list = []
    for f in csv_files:
        tmp = pd.read_csv(f)
        df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)
    
    # 2) (Optional) Only keep "growing" if that's desired
    if "fallow_label" in df.columns:
        df = df[df["fallow_label"] == "growing"].copy()
    
    # 3) Parse lat/lon and date from image_path if missing
    def parse_from_image_path(path):
        lat_match  = lat_pattern.search(path)
        lon_match  = lon_pattern.search(path)
        date_match = date_pattern.search(path)
        
        lat = float(lat_match.group(1)) if lat_match else None
        lon = float(lon_match.group(1)) if lon_match else None
        date_str = date_match.group(1) if date_match else "1900-01"
        return lat, lon, date_str
    
    # If latitude/longitude columns aren't there, parse them. Always parse date_str
    if "latitude" not in df.columns or "longitude" not in df.columns:
        df[["latitude", "longitude", "date_str"]] = df["image_path"].apply(
            lambda x: pd.Series(parse_from_image_path(x))
        )
    else:
        df["date_str"] = df["image_path"].apply(
            lambda x: parse_from_image_path(x)[2]
        )
    
    # 4) Convert date_str -> datetime, then extract year/month
    df["date_inferred"] = pd.to_datetime(df["date_str"], format="%Y-%m", errors="coerce")
    df["year_inferred"]  = df["date_inferred"].dt.year
    df["month_inferred"] = df["date_inferred"].dt.month
    
    # 5) Define function that assigns season + "season_year"
    def classify_season(row):
        m = row["month_inferred"]
        y = row["year_inferred"]
        
        # Kharif: months 7..11 (same year)
        if m in [7, 8, 9, 10, 11]:
            return "Kharif", y
        
        # Rabi: month 12 belongs to next year's Rabi
        if m == 12:
            return "Rabi", y + 1
        
        # Rabi: months 1..5 belong to the same calendar year
        if m in [1, 2, 3, 4, 5]:
            return "Rabi", y
        
        # Otherwise: "Other" (e.g., months 6, 6??)
        return "Other", y
    
    df[["season_inferred", "season_year"]] = df.apply(
        lambda r: pd.Series(classify_season(r)), axis=1
    )
    
    # 6) Drop any rows missing lat/lon
    df.dropna(subset=["latitude", "longitude"], inplace=True)
    
    return df

def load_states_shapefile(shapefile_path):
    """
    Loads the states shapefile and reprojects to an equal-area projection.
    Returns a GeoDataFrame with a column 'area_ea' storing the area in sq. meters.
    """
    states_gdf = gpd.read_file(shapefile_path)
    # Reproject to an equal-area projection for Asia/India, e.g. ESRI:102028 (Asia North Albers)
    ea_crs = "ESRI:102028" 
    states_gdf_ea = states_gdf.to_crs(ea_crs)
    
    # Calculate area in m^2
    states_gdf_ea["area_ea"] = states_gdf_ea.geometry.area
    
    # Keep area in the original GDF
    states_gdf["area_ea"] = states_gdf_ea["area_ea"]
    
    return states_gdf

def load_random_sample_csv(random_csv_path):
    """
    Loads the IndiaRandomSample2M.csv, extracts latitude & longitude from the .geo column,
    returns a GeoDataFrame in EPSG:4326 with columns ['geometry'] + the rest.
    """
    df_rand = pd.read_csv(random_csv_path)
    
    def parse_geo(geo_str):
        data = json.loads(geo_str)
        coords = data["coordinates"]  # [lon, lat]
        return Point(coords[0], coords[1])
    
    df_rand["geometry"] = df_rand[".geo"].apply(parse_geo)
    gdf_rand = gpd.GeoDataFrame(df_rand, geometry="geometry", crs="EPSG:4326")
    
    return gdf_rand

def match_inferred_fallow(
    inferred_folder, 
    shapefile_path, 
    random_csv_path,
    total_n=1000,
    out_csv="matched_inferred_fallow.csv",
    year_needed=None,
    season_needed=None,
    existing_batches=None,
    min_distance_meters=50,
    max_points_per_state=None,  # Parameter to limit points per state
    report_only=False  # If True, just report allocations without sampling
    ):
    """
    Main driver function:
      1) Load inferred_fallow points & parse year/month/season.
      2) If year_needed or season_needed are provided, filter the data.
      3) Load states shapefile (compute area)
      4) Load random sample points, spatially join with states
      5) Determine how many random points to sample per state (area-weighted)
      6) For each random point, find nearest inferred_fallow point (no repeats)
      7) Save matched inferred_fallow points to CSV (with matched_random coords).
    """
    # 1) Load the data
    df_inferred = load_inferred_fallow(inferred_folder)
  
    print(f"Loaded inferred_fallow: {len(df_inferred)} rows initially.")
    
    # Load existing matches to check for proximity
    existing_match_coords = []
    if existing_batches:
        df_inferred = remove_existing_matches(df_inferred, existing_batches)
        print(f"After removing existing matches: {len(df_inferred)} rows remain.")
        
        # Also load coordinates from existing batches for distance checking
        for batch_file in existing_batches:
            if os.path.exists(batch_file):
                batch_df = pd.read_csv(batch_file)
                # Add both original points and matched random points
                for _, row in batch_df.iterrows():
                    existing_match_coords.append([row['longitude'], row['latitude']])
                    if 'matched_random_lon' in batch_df.columns:
                        existing_match_coords.append([row['matched_random_lon'], row['matched_random_lat']])
        
        existing_match_coords = np.array(existing_match_coords)
        print(f"Loaded {len(existing_match_coords)} coordinates from existing matches for proximity checking.")

    # 2) Filter by year and/or season if specified
    if year_needed is not None:
        df_inferred = df_inferred[df_inferred['year_inferred'] == year_needed]
        print(f"After filtering by year={year_needed}, {len(df_inferred)} rows remain.")
    
    if season_needed is not None:
        df_inferred = df_inferred[df_inferred['season_inferred'] == season_needed]
        print(f"After filtering by season='{season_needed}', {len(df_inferred)} rows remain.")
    
    if df_inferred.empty:
        print("No inferred_fallow rows left after filtering. Aborting.")
        return
    
    # Convert df_inferred to a geodataframe for spatial operations
    gdf_inferred = gpd.GeoDataFrame(
        df_inferred, 
        geometry=gpd.points_from_xy(df_inferred.longitude, df_inferred.latitude),
        crs="EPSG:4326"
    )
    
    states_gdf = load_states_shapefile(shapefile_path)
    print(f"Loaded states shapefile: {len(states_gdf)} states.")
    
    # Spatially join inferred points to states to count available points per state
    if states_gdf.crs != "EPSG:4326":
        states_gdf = states_gdf.to_crs("EPSG:4326")
    
    # Join inferred points to states
    gdf_inferred = gpd.sjoin(gdf_inferred, states_gdf, how="left", predicate="intersects")
    
    # Count available points per state
    points_per_state = gdf_inferred.groupby("name_1").size()
    print(f"Available inferred points per state:")
    
    # Get all states from the shapefile for complete reporting
    all_states = states_gdf["name_1"].unique()
    
    # Report points for ALL states, not just those with points
    for state in sorted(all_states):
        count = points_per_state.get(state, 0)
        status = ""
        if count == 0:
            status = " [NO POINTS]"
        print(f"  {state}: {count}{status}")
    
    gdf_rand = load_random_sample_csv(random_csv_path)
    print(f"Loaded random sample CSV: {len(gdf_rand)} rows.")
    
    # 3) Spatially join random points to states
    gdf_rand = gpd.sjoin(gdf_rand, states_gdf, how="left", predicate="intersects")
    
    # Remove any points not matched to a state
    gdf_rand.dropna(subset=["area_ea"], inplace=True)
    
    # 4) Calculate target points per state based strictly on area
    area_sum = states_gdf["area_ea"].sum()
    
    # First pass: Strict area-based allocation
    area_allocations = {}
    for _, row in states_gdf.iterrows():
        state_name = row["name_1"]
        area_fraction = row["area_ea"] / area_sum
        
        # Calculate target strictly based on area proportion
        target_count = max(1, int(round(area_fraction * total_n)))
        area_allocations[state_name] = target_count
    
    # Adjust strict area allocations to match total_n exactly
    area_allocated_total = sum(area_allocations.values())
    if area_allocated_total != total_n:
        # Sort states by fractional part of allocation (for fair rounding)
        area_fractions = {state: row["area_ea"] / area_sum * total_n 
                          for state, row in states_gdf.iterrows()}
        
        # Calculate fractional parts
        frac_parts = {state: area_fractions[state] - int(area_fractions[state]) 
                     for state in area_fractions}
        
        if area_allocated_total < total_n:
            # Add points to states with largest fractional parts
            states_to_adjust = sorted(frac_parts.items(), key=lambda x: x[1], reverse=True)
            remaining = total_n - area_allocated_total
            
            for state, _ in states_to_adjust:
                if remaining <= 0:
                    break
                area_allocations[state] += 1
                remaining -= 1
                
        elif area_allocated_total > total_n:
            # Remove points from states with smallest fractional parts
            states_to_adjust = sorted(frac_parts.items(), key=lambda x: x[1])
            excess = area_allocated_total - total_n
            
            for state, _ in states_to_adjust:
                if excess <= 0:
                    break
                # Make sure the state exists in area_allocations before accessing
                if state in area_allocations and area_allocations[state] > 1:
                    area_allocations[state] -= 1
                    excess -= 1
    
    # Second pass: Deal with availability constraints
    state_allocations = {}
    unavailable_points = 0
    
    for state, target_count in area_allocations.items():
        available_points = points_per_state.get(state, 0)
        
        # If max_points_per_state is specified, apply that limit
        if max_points_per_state is not None:
            target_count = min(target_count, max_points_per_state)
        
        # Record how many points we'll actually be able to allocate
        actual_count = min(target_count, available_points)
        state_allocations[state] = actual_count
        
        # Track shortfall for redistribution
        if actual_count < target_count:
            unavailable_points += (target_count - actual_count)
    
    # Redistribute unavailable points to maintain area proportionality
    if unavailable_points > 0:
        print(f"\nRedistributing {unavailable_points} points due to availability constraints")
        
        # Calculate which states have extra capacity relative to area allocation
        capacity_by_state = {}
        for state in state_allocations:
            area_target = area_allocations[state]
            current = state_allocations[state]
            available = points_per_state.get(state, 0)
            
            # Extra capacity = how many more points this state could take
            extra_capacity = available - current
            
            # Only consider states that haven't hit max_points_per_state
            if max_points_per_state is not None and current >= max_points_per_state:
                extra_capacity = 0
                
            if extra_capacity > 0:
                # Weight by area to maintain proportionality
                area_fraction = states_gdf.loc[states_gdf['name_1'] == state, 'area_ea'].values[0] / area_sum
                capacity_by_state[state] = (extra_capacity, area_fraction)
        
        # Sort states by area fraction (to prioritize larger states)
        states_with_capacity = sorted(capacity_by_state.items(), key=lambda x: x[1][1], reverse=True)
        
        # Redistribute points
        points_to_add = unavailable_points
        while points_to_add > 0 and states_with_capacity:
            # Distribute one point per state per round, starting with largest area states
            added_in_round = 0
            for state, (capacity, _) in list(states_with_capacity):
                if points_to_add <= 0 or capacity <= 0:
                    continue
                    
                state_allocations[state] += 1
                points_to_add -= 1
                added_in_round += 1
                
                # Update remaining capacity
                capacity_by_state[state] = (capacity - 1, capacity_by_state[state][1])
                if capacity_by_state[state][0] <= 0:
                    states_with_capacity.remove((state, (capacity, capacity_by_state[state][1])))
            
            # If we couldn't add any points in this round, break
            if added_in_round == 0:
                break
    
    # Print summary of allocations
    print("\nArea-based allocation summary:")
    total_allocated = sum(area_allocations.values())
    for state, count in sorted(area_allocations.items(), key=lambda x: x[1], reverse=True):
        area_percent = states_gdf.loc[states_gdf['name_1'] == state, 'area_ea'].values[0] / area_sum * 100
        print(f"  {state}: {count} points ({area_percent:.1f}% of total area)")
    
    # Print states with no available points
    states_with_no_points = []
    for state in area_allocations:
        if state not in points_per_state or points_per_state.get(state, 0) == 0:
            states_with_no_points.append(state)
    
    if states_with_no_points:
        print("\nWARNING: The following states have NO AVAILABLE POINTS:")
        for state in states_with_no_points:
            area_percent = states_gdf.loc[states_gdf['name_1'] == state, 'area_ea'].values[0] / area_sum * 100
            target = area_allocations.get(state, 0)
            print(f"  {state}: Should have {target} points ({area_percent:.1f}% of area)")
    
    print("\nFinal state allocations (after availability adjustments):")
    final_allocated = sum(state_allocations.values())
    for state, count in sorted(state_allocations.items(), key=lambda x: x[1], reverse=True):
        area_percent = states_gdf.loc[states_gdf['name_1'] == state, 'area_ea'].values[0] / area_sum * 100
        allocation_percent = (count / final_allocated * 100) if final_allocated > 0 else 0
        available = points_per_state.get(state, 0)
        status = ""
        if available == 0:
            status = " [NO POINTS AVAILABLE]"
        elif available < area_allocations.get(state, 0):
            status = f" [INSUFFICIENT: {available}/{area_allocations.get(state, 0)}]"
        
        print(f"  {state}: {count} points ({allocation_percent:.1f}% of allocation, {area_percent:.1f}% of area, {available} available){status}")
    
    # If report_only flag is set, stop here
    if report_only:
        print("\nReport-only mode: Stopping before sampling")
        return
    
    # 5) Process states sequentially to match the target allocations
    matched_rows = []
    
    # Group inferred points and random points by state
    inferred_by_state = {state: gdf_inferred[gdf_inferred['name_1'] == state] 
                         for state in state_allocations.keys()}
    
    random_by_state = {state: gdf_rand[gdf_rand['name_1'] == state] 
                       for state in state_allocations.keys()}
    
    # Track matched coordinates for minimum distance checking
    all_matched_coords = np.array(existing_match_coords) if len(existing_match_coords) > 0 else np.empty((0, 2))
    
    # Process each state
    total_matched = 0
    
    for state, target_count in state_allocations.items():
        if target_count <= 0:
            continue
            
        print(f"\nProcessing state: {state}, target: {target_count}")
        
        state_inferred = inferred_by_state.get(state)
        state_random = random_by_state.get(state)
        
        if state_inferred is None or state_random is None or len(state_inferred) == 0 or len(state_random) == 0:
            print(f"  Skipping {state} - no points available")
            continue
        
        # Convert to numpy arrays for KDTree
        inferred_coords = state_inferred[["longitude", "latitude"]].values
        inferred_indices = state_inferred.index.values
        
        # Shuffle random points to avoid spatial clustering
        random_indices = random.sample(list(state_random.index), min(len(state_random), target_count * 3))
        
        state_matched = 0
        state_matched_coords = []
        
        # Build KDTree for this state's inferred points
        kdtree = cKDTree(inferred_coords)
        
        for rand_idx in random_indices:
            if state_matched >= target_count:
                break
                
            row_rnd = state_random.loc[rand_idx]
            rnd_lon = row_rnd.geometry.x
            rnd_lat = row_rnd.geometry.y
            
            # Get k nearest neighbors to find one that meets distance requirement
            k = min(10, len(inferred_coords))
            if k == 0:
                break
                
            distances, indices = kdtree.query([rnd_lon, rnd_lat], k=k)
            
            # Convert to list if single value
            if k == 1:
                distances = [distances]
                indices = [indices]
            
            # Try each nearby point until finding one that meets distance requirement
            for dist, idx_kd in zip(distances, indices):
                matched_df_idx = inferred_indices[idx_kd]
                matched_row = df_inferred.loc[matched_df_idx]
                
                point_coords = np.array([matched_row['longitude'], matched_row['latitude']])
                
                # Check minimum distance against all matched points
                if check_min_distance(point_coords, all_matched_coords, min_distance_meters):
                    # Point meets distance requirement - use it
                    matched_row = matched_row.copy()
                    matched_row["matched_random_lon"] = rnd_lon
                    matched_row["matched_random_lat"] = rnd_lat
                    matched_row["state"] = state  # Add state information
                    
                    matched_rows.append(matched_row)
                    
                    # Update coordinates for distance checking
                    state_matched_coords.append(point_coords)
                    if len(all_matched_coords) == 0:
                        all_matched_coords = np.array([point_coords])
                    else:
                        all_matched_coords = np.vstack([all_matched_coords, point_coords])
                    
                    # Remove this point from consideration
                    mask = (inferred_coords != point_coords).any(axis=1)
                    inferred_coords = inferred_coords[mask]
                    inferred_indices = inferred_indices[mask]
                    
                    state_matched += 1
                    total_matched += 1
                    break
            
            # Rebuild KDTree after removing points
            if len(inferred_coords) > 0:
                kdtree = cKDTree(inferred_coords)
            else:
                break
        
        print(f"  Matched {state_matched} out of {target_count} target points for {state}")
    
    # 6) Create and save final DataFrame
    df_matched = pd.DataFrame(matched_rows)
    
    # Count matches by state
    state_counts = df_matched['state'].value_counts()
    print("\nFinal sampling distribution by state:")
    for state, count in state_counts.items():
        print(f"  {state}: {count}")
    
    # Summary statistics at the end
    print(f"\nSampling Results Summary:")
    print(f"- Successfully matched {len(df_matched)} points (out of {total_n} desired)")
    
    # Count states with zero matches
    states_with_matches = set(df_matched['state']) if 'state' in df_matched.columns else set()
    states_with_allocations = set(state_allocations.keys())
    states_missing = states_with_allocations - states_with_matches
    
    if states_missing:
        print(f"- WARNING: {len(states_missing)} states received zero matches despite allocations:")
        for state in sorted(states_missing):
            print(f"  * {state}: allocated {state_allocations.get(state, 0)} points but matched 0")
    
    # Count states with fewer matches than allocated
    under_allocated = []
    for state, count in state_allocations.items():
        if state in states_with_matches:
            actual = df_matched['state'].value_counts().get(state, 0)
            if actual < count:
                under_allocated.append((state, actual, count))
    
    if under_allocated:
        print(f"- {len(under_allocated)} states received fewer points than allocated:")
        for state, actual, allocated in sorted(under_allocated, key=lambda x: (x[1]/x[2], x[2]), reverse=False):
            print(f"  * {state}: matched {actual}/{allocated} points ({actual/allocated*100:.1f}%)")
            
    # Print final distribution vs area comparison
    print("\nFinal distribution compared to area proportion:")
    total_matched = len(df_matched)
    
    state_counts = df_matched['state'].value_counts() if 'state' in df_matched.columns else pd.Series()
    for state in sorted(state_counts.index):
        count = state_counts.get(state, 0)
        area_percent = states_gdf.loc[states_gdf['name_1'] == state, 'area_ea'].values[0] / area_sum * 100
        match_percent = (count / total_matched * 100) if total_matched > 0 else 0
        diff = match_percent - area_percent
        
        # Add indicator for significant deviation
        status = ""
        if abs(diff) > 3:  # More than 3% difference
            if diff > 0:
                status = " [OVERREPRESENTED]"
            else:
                status = " [UNDERREPRESENTED]"
        
        print(f"  {state}: {count} points ({match_percent:.1f}% of sample vs {area_percent:.1f}% of area, diff: {diff:+.1f}%){status}")
    
    df_matched.to_csv(out_csv, index=False)
    print(f"Saved matched inferred_fallow points to: {out_csv}")

# ========================
# Example usage
# ========================
if __name__ == "__main__":
    out_csv         = "/home/laguarta_jordi/sean7391/streetview_highres/stratifiedSampleMatching_6k_2023_Kharif.csv"
    random_csv_path = "/home/laguarta_jordi/sean7391/streetview_highres/IndiaRandomSample2M.csv"

    inferred_folder = "/home/laguarta_jordi/sean7391/inferred_fallow"
    shapefile_path  = "/home/laguarta_jordi/sean7391/SecondLevelIndiaShp/secondLevelIndiaShp.shp"


    existing_batches = [
        "/home/laguarta_jordi/sean7391/streetview_highres/stratifiedSampleMatching_3k_2023_Kharif.csv",
        # "previous_batch_2.csv"
    ]

    # For example, we want 2000 total matches from the year 2023, season Kharif
    match_inferred_fallow(
        inferred_folder=inferred_folder,
        shapefile_path=shapefile_path,
        random_csv_path=random_csv_path,
        total_n=3000,
        out_csv=out_csv,
        year_needed=2023,         # Filter to year_inferred == 2023
        season_needed="Kharif",   # Filter to season_inferred == "Kharif"
        existing_batches=existing_batches,
        min_distance_meters=100,  
        max_points_per_state=600  # Limit max points per state (can be None for no limit)
    )
    
    # Alternative: Run with a report-only flag to just see allocations without sampling
    """
    match_inferred_fallow(
        inferred_folder=inferred_folder,
        shapefile_path=shapefile_path,
        random_csv_path=random_csv_path,
        total_n=2000,
        out_csv=None,  # Set to None to run in report-only mode
        year_needed=2023,
        season_needed="Kharif",
        existing_batches=existing_batches,
        report_only=True  # Just print the allocations without sampling
    )
    """