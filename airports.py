#!/usr/bin/env python3
"""
Airport data and lookup functionality for flight tracker.
Uses airports.csv file to provide comprehensive airport data.
"""
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Default path for airport data
DEFAULT_AIRPORT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airports.csv")

# Types of airports we're interested in (can be adjusted)
VALID_AIRPORT_TYPES = ["large_airport", "medium_airport", "small_airport"]

# Airport data caches
_airport_data = {}  # All loaded airport data
_iata_to_idx = {}   # IATA code lookup
_icao_to_idx = {}   # ICAO code lookup

def load_airport_data(csv_path: str = DEFAULT_AIRPORT_CSV) -> bool:
    """
    Load airport data from CSV file.
    
    Args:
        csv_path (str): Path to the airport CSV file
        
    Returns:
        bool: True if data was loaded successfully, False otherwise
    """
    global _airport_data, _iata_to_idx, _icao_to_idx
    
    # Reset data
    _airport_data = {}
    _iata_to_idx = {}
    _icao_to_idx = {}
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Airport data file not found: {csv_path}")
        return False
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=',')
            for idx, row in enumerate(reader):
                # Store all airport data
                _airport_data[idx] = row
                
                # Create lookups for IATA and ICAO codes (only if they exist)
                if row.get('iata_code') and row['iata_code'].strip():
                    _iata_to_idx[row['iata_code'].strip().upper()] = idx
                    
                if row.get('icao_code') and row['icao_code'].strip():
                    _icao_to_idx[row['icao_code'].strip().upper()] = idx
        
        print(f"Loaded {len(_airport_data)} airports from {csv_path}")
        print(f"Found {len(_iata_to_idx)} airports with IATA codes and {len(_icao_to_idx)} with ICAO codes")
        return True
    
    except Exception as e:
        print(f"Error loading airport data: {str(e)}")
        return False

def get_airport_coordinates(code: str) -> Optional[Tuple[float, float]]:
    """
    Get the coordinates of an airport by IATA or ICAO code.
    
    Args:
        code (str): The IATA or ICAO code of the airport
    
    Returns:
        tuple: (latitude, longitude) if found, None otherwise
    """
    # Load data if not already loaded
    if not _airport_data:
        if not load_airport_data():
            return None
    
    code = code.strip().upper()
    
    # Try IATA lookup
    if code in _iata_to_idx:
        airport = _airport_data[_iata_to_idx[code]]
        try:
            lat = float(airport['latitude_deg'])
            lon = float(airport['longitude_deg'])
            return lat, lon
        except (ValueError, KeyError):
            pass
    
    # Try ICAO lookup
    if code in _icao_to_idx:
        airport = _airport_data[_icao_to_idx[code]]
        try:
            lat = float(airport['latitude_deg'])
            lon = float(airport['longitude_deg'])
            return lat, lon
        except (ValueError, KeyError):
            pass
    
    return None

def get_airport_info(code: str) -> Optional[Dict[str, Any]]:
    """
    Get full airport information by IATA or ICAO code.
    
    Args:
        code (str): The IATA or ICAO code of the airport
    
    Returns:
        dict: Airport information if found, None otherwise
    """
    # Load data if not already loaded
    if not _airport_data:
        if not load_airport_data():
            return None
    
    code = code.strip().upper()
    airport_idx = None
    
    # Try IATA lookup
    if code in _iata_to_idx:
        airport_idx = _iata_to_idx[code]
    
    # Try ICAO lookup
    elif code in _icao_to_idx:
        airport_idx = _icao_to_idx[code]
    
    # Return None if not found
    if airport_idx is None:
        return None
    
    # Get the airport data
    airport = _airport_data[airport_idx]
    
    # Convert to our standardized format
    try:
        return {
            'iata': airport.get('iata_code', '').strip().upper() or None,
            'icao': airport.get('icao_code', '').strip().upper() or None,
            'latitude': float(airport['latitude_deg']),
            'longitude': float(airport['longitude_deg']),
            'name': airport.get('name', ''),
            'city': airport.get('municipality', ''),
            'country': airport.get('iso_country', ''),
            'elevation_ft': int(float(airport['elevation_ft'])) if airport.get('elevation_ft') else None,
            'type': airport.get('type', '')
        }
    except (ValueError, KeyError) as e:
        print(f"Error processing airport {code}: {str(e)}")
        return None

def get_airport_bounding_box(code: str, radius_km: float = 20) -> Optional[Tuple[float, float, float, float]]:
    """
    Get a bounding box around an airport based on IATA or ICAO code.
    
    Args:
        code (str): The IATA or ICAO code of the airport
        radius_km (float): Radius around airport in kilometers (default: 20km)
    
    Returns:
        tuple: (min_lat, max_lat, min_lon, max_lon) if found, None otherwise
    """
    coords = get_airport_coordinates(code)
    if not coords:
        return None
    
    lat, lon = coords
    
    # Approximate conversion: 1 degree of latitude = ~111km
    # Longitude degrees vary with latitude, so use cos(lat)
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * abs(math.cos(math.radians(lat))))
    
    min_lat = lat - lat_delta
    max_lat = lat + lat_delta
    min_lon = lon - lon_delta
    max_lon = lon + lon_delta
    
    return min_lat, max_lat, min_lon, max_lon

def search_airports(query: str, search_fields: List[str] = None) -> List[Dict[str, Any]]:
    """
    Search for airports by name, city, country, IATA, or ICAO code.
    
    Args:
        query (str): The search query
        search_fields (list): List of fields to search in (default: name, municipality, iso_country, iata_code, icao_code)
    
    Returns:
        list: List of matching airport dictionaries
    """
    # Load data if not already loaded
    if not _airport_data:
        if not load_airport_data():
            return []
    
    if not search_fields:
        search_fields = ['name', 'municipality', 'iso_country', 'iata_code', 'icao_code']
    
    query = query.lower()
    results = []
    
    for idx, airport in _airport_data.items():
        # Check if any of the search fields match the query
        for field in search_fields:
            if field in airport and airport[field] and query in airport[field].lower():
                # Convert to our standardized format
                try:
                    results.append({
                        'iata': airport.get('iata_code', '').strip().upper() or None,
                        'icao': airport.get('icao_code', '').strip().upper() or None,
                        'latitude': float(airport['latitude_deg']),
                        'longitude': float(airport['longitude_deg']),
                        'name': airport.get('name', ''),
                        'city': airport.get('municipality', ''),
                        'country': airport.get('iso_country', ''),
                        'elevation_ft': int(float(airport['elevation_ft'])) if airport.get('elevation_ft') else None,
                        'type': airport.get('type', '')
                    })
                    break  # Don't add the same airport multiple times
                except (ValueError, KeyError):
                    pass
    
    return results

def list_airports(limit: int = 100, major_only: bool = True) -> str:
    """
    Return a formatted list of available airports.
    
    Args:
        limit (int): Maximum number of airports to list
        major_only (bool): If True, only include major airports
        
    Returns:
        str: Formatted string with airport information
    """
    # Load data if not already loaded
    if not _airport_data:
        if not load_airport_data():
            return "Error: Failed to load airport data."
    
    result = []
    result.append("Available airports:")
    result.append("IATA | ICAO | Airport Name | Location | Type")
    result.append("-" * 70)
    
    count = 0
    
    # Get all IATA codes (sorted)
    iata_codes = sorted(_iata_to_idx.keys())
    
    for iata in iata_codes:
        idx = _iata_to_idx[iata]
        airport = _airport_data[idx]
        
        # Skip non-major airports if requested
        if major_only and airport.get('type') not in VALID_AIRPORT_TYPES:
            continue
        
        # Format the airport data
        airport_type = airport.get('type', '').replace('_', ' ').title()
        airport_name = airport.get('name', '')
        icao = airport.get('icao_code', '').strip().upper()
        city = airport.get('municipality', '')
        country = airport.get('iso_country', '')
        
        result.append(f"{iata} | {icao} | {airport_name} | {city}, {country} | {airport_type}")
        
        count += 1
        if count >= limit:
            result.append(f"\n... and {len(_iata_to_idx) - limit} more airports (showing {limit} of {len(_iata_to_idx)})")
            break
    
    return "\n".join(result)

def check_airport_data_file() -> str:
    """
    Check if the airport data file exists and return status message.
    
    Returns:
        str: Status message about airport data
    """
    if os.path.exists(DEFAULT_AIRPORT_CSV):
        file_size = os.path.getsize(DEFAULT_AIRPORT_CSV) / (1024 * 1024)  # Size in MB
        return f"Airport data file found: {DEFAULT_AIRPORT_CSV} ({file_size:.1f} MB)"
    else:
        return f"Airport data file not found: {DEFAULT_AIRPORT_CSV}. Please download it from https://ourairports.com/data/"

# Initialize data when module is imported
if __name__ != "__main__":
    # Try to load data but don't fail if file doesn't exist yet
    load_airport_data()

if __name__ == "__main__":
    # If the script is run directly, show airport data info and sample list
    print(check_airport_data_file())
    
    # Try to load data
    if load_airport_data():
        # Print some stats
        print("\nAirport Database Statistics:")
        print(f"Total airports: {len(_airport_data)}")
        print(f"Airports with IATA codes: {len(_iata_to_idx)}")
        print(f"Airports with ICAO codes: {len(_icao_to_idx)}")
        
        # Print a sample of major airports
        print("\n" + list_airports(limit=20, major_only=True))
        
        # Example usage
        print("\nExample lookups:")
        for code in ["JFK", "KJFK", "LHR", "EGLL"]:
            info = get_airport_info(code)
            if info:
                print(f"{code}: {info['name']} in {info['city']}, {info['country']} at ({info['latitude']}, {info['longitude']})")
            else:
                print(f"{code}: Not found")
    else:
        print("\nFailed to load airport data.")
        print("Please download the airports.csv file from https://ourairports.com/data/")
        print(f"and place it at {DEFAULT_AIRPORT_CSV}")
