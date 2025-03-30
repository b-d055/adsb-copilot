#!/usr/bin/env python3
import curses
import json
import math
import argparse
import time
import requests
from datetime import datetime
import threading
import os
import sys

# Import from our airports module
try:
    from airports import get_airport_bounding_box, get_airport_info, list_airports
    from logger import get_logger
except ImportError:
    # If the script is run without the module in path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from airports import get_airport_bounding_box, get_airport_info, list_airports
        from logger import get_logger
    except ImportError:
        print("Error: Could not import required modules. Make sure airports.py and logger.py are in the same directory.")
        sys.exit(1)

class FlightTracker:
    def __init__(self, stdscr, args):
        self.stdscr = stdscr
        self.data = None
        self.previous_valid_data = None  # Store last valid data to prevent "No flight data" flashes
        self.refreshing_data = False  # Flag to indicate data is being refreshed
        self.min_lat = args.min_lat
        self.max_lat = args.max_lat
        self.min_lon = args.min_lon
        self.max_lon = args.max_lon
        self.refresh_rate = args.refresh
        self.selected_idx = 0
        self.selected_callsign = None  # Store selected aircraft callsign for consistent selection
        self.offset = 0
        self.map_height = 0
        self.map_width = 0
        self.status_message = "Starting..."
        self.last_update_time = 0
        self.update_thread = None
        self.running = True
        self.paused = False  # Add pause state variable
        self.username = args.username
        self.password = args.password
        self.lock = threading.Lock()
        self.airport_info = None
        
        # LLM integration
        self.use_llm = args.use_llm  # Flag to enable/disable LLM
        self.ollama_url = args.ollama_url  # URL for Ollama server
        self.ollama_model = args.ollama_model  # Model to use
        self.llm_refresh_rate = args.llm_refresh  # LLM refresh rate in seconds
        self.aircraft_summary = None  # Store the summary from LLM
        self.summary_callsign = None  # Track which aircraft summary is for
        self.summary_thread = None  # Thread for fetching summaries
        self.summary_lock = threading.Lock()  # Lock for accessing summary data
        self.is_fetching_summary = False  # Flag to track summary fetching status
        
        # Initialize logger
        self.logger = get_logger(args.log_file, args.enable_logging)
        self.logger.info(f"Flight Tracker started with bounds: lat({self.min_lat},{self.max_lat}), lon({self.min_lon},{self.max_lon})")
        self.logger.info(f"Refresh rate: {self.refresh_rate}s, API Auth: {'Yes' if self.username else 'No'}")
        
        if self.use_llm:
            self.logger.info(f"LLM integration enabled: {self.ollama_url}, model: {self.ollama_model}, refresh: {self.llm_refresh_rate}s")
        
        # Dictionary to track the latest position timestamp for each aircraft
        # Format: {icao24: {'time_position': timestamp, 'last_seen': timestamp}}
        self.aircraft_timestamps = {}
        
        # Store airport information if provided
        if args.airport:
            self.airport_info = get_airport_info(args.airport)
            if self.airport_info:
                # Update the bounds
                self.min_lat, self.max_lat, self.min_lon, self.max_lon = get_airport_bounding_box(
                    args.airport, args.radius
                )
                airport_name = self.airport_info['name']
                self.status_message = f"Tracking flights around {airport_name} ({args.airport})"
                self.logger.info(f"Using airport: {airport_name} ({args.airport}) with radius {args.radius}km")
                self.logger.info(f"Updated bounds: lat({self.min_lat},{self.max_lat}), lon({self.min_lon},{self.max_lon})")
        
        # Initialize data
        self.fetch_data()
    
    def fetch_data(self):
        """Fetch flight data from OpenSky Network API"""
        # If paused, don't fetch new data
        if self.paused:
            return
            
        # Set the refreshing flag
        self.refreshing_data = True
        
        # Construct the API URL with bounding box
        url = f"https://opensky-network.org/api/states/all?lamin={self.min_lat}&lomin={self.min_lon}&lamax={self.max_lat}&lomax={self.max_lon}"
        
        # Setup authentication if provided
        auth = None
        if self.username and self.password:
            auth = (self.username, self.password)
        
        self.logger.info(f"Auth {auth}")
        # Log the request
        self.logger.debug(f"Fetching data from API: {url}")
        
        try:
            # Make the API request
            start_time = time.time()
            response = requests.get(url, auth=auth, timeout=10)
            
            # Check for rate limiting before raising for status
            if response.status_code == 429:
                retry_after = response.headers.get('X-Rate-Limit-Retry-After-Seconds', 'unknown')
                error_msg = f"Rate limited by OpenSky API. Retry after: {retry_after} seconds"
                self.logger.warning(error_msg)
                self.status_message = f"API Rate Limited. Retry after: {retry_after}s"
                # Log the request with the rate limit information
                self.logger.log_request(url, response.status_code, time.time() - start_time, 
                                       error=f"Rate limited. Retry after: {retry_after}s")
                return  # Exit early
            
            response.raise_for_status()  # Raise exception for other 4XX/5XX responses
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Log the successful response
            self.logger.log_request(url, response.status_code, duration)
            
            # Parse the JSON response
            with self.lock:
                raw_data = response.json()
                
                # Check if we have valid states data
                if 'states' not in raw_data or raw_data['states'] is None:
                    self.logger.warning("No valid flight states in API response")
                    raise ValueError("No valid flight states in API response")
                
                # Process the states to filter out old positions
                if raw_data['states']:
                    filtered_states = []
                    stale_positions = 0  # Track how many positions were filtered out
                    outdated_but_kept = 0  # Track positions that are outdated but kept (< 1 minute old)
                    current_time = time.time()
                    
                    for state in raw_data['states']:
                        # Get the ICAO24 identifier
                        icao24 = state[0] if state[0] else None
                        
                        if not icao24:
                            # Skip aircraft without ICAO24 identifier
                            continue
                        
                        # Get position time and last contact time
                        time_position = state[3]  # Index 3 is the position timestamp
                        last_contact = state[4]   # Index 4 is the last contact timestamp
                        
                        # Only filter positions older than 1 minute (60 seconds)
                        if time_position is None:
                            # Skip aircraft with no position data
                            continue
                            
                        position_age = current_time - time_position
                        
                        # Check if we have seen this aircraft before
                        if icao24 in self.aircraft_timestamps:
                            last_known_position_time = self.aircraft_timestamps[icao24]['time_position']
                            
                            # If the position is older than 1 minute, skip it
                            if position_age > 60:
                                stale_positions += 1
                                continue
                                
                            # If new position is older than what we have, but still within 1 minute
                            if last_known_position_time is not None and time_position <= last_known_position_time:
                                outdated_but_kept += 1
                        
                        # Update our tracking of this aircraft's timestamps
                        self.aircraft_timestamps[icao24] = {
                            'time_position': time_position,
                            'last_seen': last_contact
                        }
                        
                        # Keep this state in our filtered list
                        filtered_states.append(state)
                    
                    # Replace the original states with our filtered list
                    raw_data['states'] = filtered_states
                    
                    # Log the data update stats
                    self.logger.log_data_update({
                        "total_aircraft": len(raw_data['states']),
                        "filtered_stale": stale_positions,
                        "outdated_but_kept": outdated_but_kept,
                        "tracked_unique": len(self.aircraft_timestamps),
                        "request_time": f"{duration:.2f}s",
                        "bounds": f"lat({self.min_lat:.2f},{self.max_lat:.2f}),lon({self.min_lon:.2f},{self.max_lon:.2f})"
                    })
                    
                    # Update status message with filtering info
                    if stale_positions > 0 or outdated_but_kept > 0:
                        self.status_message = f"Updated: {len(filtered_states)} aircraft. Filtered {stale_positions} stale positions (>1min). Kept {outdated_but_kept} outdated positions."
                    else:
                        self.status_message = f"Updated: {len(filtered_states)} aircraft. Request took {duration:.2f}s"
                
                # Store valid data in both current and previous slots
                if raw_data['states']:  # Only update if we have actual aircraft states
                    # Before updating data, save the callsign of the currently selected aircraft
                    currently_selected_callsign = None
                    if self.data and 'states' in self.data and self.data['states'] and 0 <= self.selected_idx < len(self.data['states']):
                        current_selection = self.data['states'][self.selected_idx]
                        if current_selection[1]:  # Callsign is at index 1
                            currently_selected_callsign = current_selection[1].strip()
                    
                    # Update the data
                    self.data = raw_data
                    self.previous_valid_data = raw_data
                    self.last_update_time = time.time()
                    
                    # Update selection based on callsign
                    if currently_selected_callsign:
                        self.selected_callsign = currently_selected_callsign
                        self.update_selection_by_callsign()
                else:
                    # If no states but we had previous data, keep using it
                    if self.previous_valid_data:
                        self.status_message = "No aircraft in selected area. Showing previous data."
                        self.logger.info("No aircraft in selected area. Using previous data.")
                    else:
                        self.status_message = "No aircraft found in the selected area."
                        self.logger.warning("No aircraft found in the selected area.")
        
        except requests.exceptions.RequestException as e:
            self.status_message = f"API Error: {str(e)}"
            # Check if this is a HTTPError with status code
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                # Additional handling for specific status codes
                if status_code == 429:  # Rate limit - should be caught above, but just in case
                    retry_after = e.response.headers.get('X-Rate-Limit-Retry-After-Seconds', 'unknown')
                    self.logger.warning(f"Rate limited by OpenSky API. Retry after: {retry_after} seconds")
                self.logger.log_request(url, status_code, error=str(e))
            else:
                self.logger.log_request(url, error=str(e))
        except json.JSONDecodeError:
            self.status_message = "Error: Invalid JSON response from API"
            self.logger.error("Invalid JSON response from API")
        except ValueError as e:
            self.status_message = str(e)
            self.logger.warning(str(e))
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.logger.error(f"Unexpected error: {str(e)}")
        finally:
            # Clear the refreshing flag
            self.refreshing_data = False
    
    def update_selection_by_callsign(self):
        """Update the selected index based on the stored callsign"""
        if not self.selected_callsign or not self.data or 'states' not in self.data or not self.data['states']:
            self.selected_idx = 0  # Default to first aircraft if no callsign or no data
            return
            
        states = self.data['states']
        found = False
        
        for i, state in enumerate(states):
            if state[1] and state[1].strip() == self.selected_callsign:
                self.selected_idx = i
                found = True
                break
        
        if not found:
            # If the aircraft is no longer in the data, select the first one
            self.selected_idx = 0
            # Update the callsign to the new selection if available
            if states and len(states) > 0 and states[0][1]:
                self.selected_callsign = states[0][1].strip()
            else:
                self.selected_callsign = None
    
    def update_loop(self):
        """Background thread that updates flight data periodically"""
        while self.running:
            # Only fetch data if not paused
            if not self.paused:
                self.fetch_data()
            # Sleep for the specified refresh rate
            time.sleep(self.refresh_rate)
    
    def geo_to_screen(self, lat, lon):
        """Convert geographical coordinates to screen coordinates"""
        if lat is None or lon is None:
            return None, None
            
        # Calculate screen coordinates
        y = self.map_height - int((lat - self.min_lat) / (self.max_lat - self.min_lat) * self.map_height)
        x = int((lon - self.min_lon) / (self.max_lon - self.min_lon) * self.map_width)
        
        return x, y
    
    # Safe string writing function to prevent boundary errors
    def safe_addstr(self, y, x, string, attr=curses.A_NORMAL):
        """Safely add a string to the screen, truncating if necessary."""
        height, width = self.stdscr.getmaxyx()
        # Check if we're trying to write outside the screen
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        
        # Truncate the string if it would go beyond the right edge
        max_len = width - x
        if max_len <= 0:
            return
            
        display_str = string[:max_len]
        try:
            self.stdscr.addstr(y, x, display_str, attr)
        except curses.error:
            # Catch any remaining curses errors
            pass
    
    def draw_map(self):
        """Draw the base map and aircraft positions"""
        # Clear the screen
        self.stdscr.clear()
        
        # Get screen dimensions
        max_y, max_x = self.stdscr.getmaxyx()
        
        # Calculate layout with space for all sections
        info_panel_height = 10
        legend_height = 6
        llm_summary_height = 6 if self.use_llm else 0  # Allocate space for LLM summary if enabled
        status_bar_height = 2  # Height of the status bar at the bottom
        
        # Extra padding between sections to prevent overlap
        section_padding = 1
        
        # Calculate total space needed for bottom sections
        bottom_sections_height = (
            info_panel_height + 
            llm_summary_height + 
            legend_height + 
            (section_padding * 2) +  # Add padding between sections
            status_bar_height        # Reserve space for status bar
        )
        
        # Adjust map height to make room for all sections
        self.map_height = max(5, max_y - bottom_sections_height)
        self.map_width = max_x
        
        # Draw border around map (safely)
        for y in range(self.map_height):
            try:
                if y < max_y and 0 < max_x:
                    self.stdscr.addch(y, 0, '|')
                if y < max_y and max_x-1 < max_x:
                    self.stdscr.addch(y, max_x-1, '|')
            except curses.error:
                pass
        
        for x in range(max_x):
            try:
                if 0 < max_y and x < max_x:
                    self.stdscr.addch(0, x, '-')
                if self.map_height-1 < max_y and x < max_x:
                    self.stdscr.addch(self.map_height-1, x, '-')
            except curses.error:
                pass
        
        # Draw location grid labels
        grid_labels = [
            f"({self.min_lat:.1f}, {self.min_lon:.1f})",
            f"({self.min_lat:.1f}, {self.max_lon:.1f})",
            f"({self.max_lat:.1f}, {self.min_lon:.1f})",
            f"({self.max_lat:.1f}, {self.max_lon:.1f})"
        ]
        
        # Bottom-left corner
        self.safe_addstr(self.map_height-2, 2, grid_labels[0])
        # Bottom-right corner
        self.safe_addstr(self.map_height-2, max_x-len(grid_labels[1])-2, grid_labels[1])
        # Top-left corner
        self.safe_addstr(2, 2, grid_labels[2])
        # Top-right corner
        self.safe_addstr(2, max_x-len(grid_labels[3])-2, grid_labels[3])
        
        # Draw aircraft with the lock to prevent data race
        with self.lock:
            display_data = self.data if self.data else self.previous_valid_data
            
            if display_data and 'states' in display_data and display_data['states']:
                # Ensure the selected index is valid
                self.update_selection_by_callsign()
                
                states = display_data['states']
                for i, state in enumerate(states):
                    if i >= len(states):
                        break
                        
                    lat = state[6]  # latitude at index 6
                    lon = state[5]  # longitude at index 5
                    on_ground = state[8]  # on_ground at index 8
                    heading = state[10]  # true_track at index 10
                    
                    if lat is not None and lon is not None:
                        x, y = self.geo_to_screen(lat, lon)
                        
                        if x is not None and y is not None and 0 <= x < max_x-1 and 0 <= y < self.map_height-1:
                            # Use different symbols based on heading for all aircraft,
                            # regardless of whether they're on the ground
                            if heading is not None:
                                if 22.5 <= heading < 67.5:
                                    symbol = '↗'   # Northeast
                                elif 67.5 <= heading < 112.5:
                                    symbol = '→'   # East
                                elif 112.5 <= heading < 157.5:
                                    symbol = '↘'   # Southeast
                                elif 157.5 <= heading < 202.5:
                                    symbol = '↓'   # South
                                elif 202.5 <= heading < 247.5:
                                    symbol = '↙'   # Southwest
                                elif 247.5 <= heading < 292.5:
                                    symbol = '←'   # West
                                elif 292.5 <= heading < 337.5:
                                    symbol = '↖'   # Northwest
                                else:
                                    symbol = '↑'   # North
                            else:
                                # For aircraft with unknown heading
                                # Use a different symbol for ground vs airborne aircraft with unknown heading
                                symbol = '#' if on_ground else '•'
                            
                            try:
                                # Get the appropriate color for the aircraft
                                color_attr = self.get_aircraft_color(state)
                                
                                # Highlight selected aircraft
                                if i == self.selected_idx:
                                    self.stdscr.addch(y, x, symbol, curses.A_REVERSE | color_attr)
                                else:
                                    self.stdscr.addch(y, x, symbol, color_attr)
                            except curses.error:
                                pass
        
        # Draw information panel
        info_panel_start_y = self.map_height
        self.draw_info_panel(info_panel_start_y, max_x)
        
        # Draw LLM summary if enabled
        llm_summary_start_y = info_panel_start_y + info_panel_height + section_padding
        if self.use_llm:
            self.draw_llm_summary(llm_summary_start_y, max_x)
        
        # Draw the legend - position after info panel and LLM summary
        legend_start_y = llm_summary_start_y + (llm_summary_height if self.use_llm else 0) + section_padding
        self.draw_legend(legend_start_y, max_x)
        
        # Draw status bar (split into two lines)
        # Make sure to position the status bar at the bottom of the screen
        status_bar_y = max_y - status_bar_height
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # First line of status bar with fixed information
        status_line1 = f"Time: {current_time} | "
        
        with self.lock:
            if self.paused:
                status_line1 += "DATA UPDATES PAUSED | "
            elif self.refreshing_data:
                status_line1 += "Updating data... | "
            elif self.data:
                status_line1 += f"Last API update: {datetime.fromtimestamp(self.last_update_time).strftime('%H:%M:%S')} | "
        
        status_line1 += f"Refresh: {self.refresh_rate}s"
        
        # Second line with dynamic content (counts and status message)
        status_line2 = ""
        with self.lock:
            if self.data:
                status_line2 += f"Flights: {len(self.data.get('states', []))} | "
        
        status_line2 += f"Arrow keys:select, r:refresh, p:pause/resume, +/-:zoom, q:quit | {self.status_message}"
        
        # Safely add the status lines at the fixed position
        self.safe_addstr(status_bar_y, 0, status_line1)
        self.safe_addstr(status_bar_y + 1, 0, status_line2)
        
        # Refresh the screen
        self.stdscr.refresh()
    
    def get_aircraft_summary(self, selected_state=None):
        """Get a summary of the current airspace from the LLM"""
        # Don't attempt if LLM is not enabled or if data updates are paused
        if not self.use_llm or self.paused:
            return None
            
        # Set flag to indicate we're fetching a summary
        self.is_fetching_summary = True
        
        try:
            # Get all aircraft data
            with self.lock:
                display_data = self.data if self.data else self.previous_valid_data
                if not display_data or 'states' not in display_data or not display_data['states']:
                    return "No aircraft data available for analysis."
                
                # Get all aircraft states
                states = display_data['states']
                if not states:
                    return "No aircraft currently in the monitored airspace."
                
                # Get the selected aircraft callsign
                selected_callsign = None
                if selected_state and selected_state[1]:
                    selected_callsign = selected_state[1].strip()
            
            # Format aircraft data for the LLM prompt
            aircraft_list = []
            for i, state in enumerate(states[:min(25, len(states))]):  # Limit to 25 aircraft to keep prompt size reasonable
                icao24 = state[0] or "N/A"
                callsign = state[1].strip() if state[1] else "N/A"
                country = state[2] or "N/A"
                
                lat = state[6]
                lon = state[5]
                position = f"{lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else "N/A"
                
                baro_alt = state[7]
                altitude = f"{int(baro_alt)}m" if baro_alt is not None else "N/A"
                
                velocity = state[9]
                speed = f"{int(velocity)}m/s" if velocity is not None else "N/A"
                
                true_track = state[10]
                heading = f"{int(true_track)}°" if true_track is not None else "N/A"
                
                vert_rate = state[11]
                v_rate = f"{int(vert_rate)}m/s" if vert_rate is not None else "N/A"
                
                on_ground = state[8]
                on_ground_str = "Yes" if on_ground else "No" if on_ground is not None else "Unknown"
                
                # Append a formatted string for each aircraft
                aircraft_list.append(
                    f"Callsign {callsign}:\n"
                    f"  ICAO: {icao24}, Aircraft {i+1}, Country: {country}\n"
                    f"  Position: {position}, Altitude: {altitude}, Speed: {speed}\n"
                    f"  Heading: {heading}, Vertical Rate: {v_rate}, On Ground: {on_ground_str}"
                )
            
            total_aircraft = len(states)
            if total_aircraft > 25:
                aircraft_list.append(f"... and {total_aircraft - 25} more aircraft")
                
            # Build the complete prompt
            aircraft_data = "\n".join(aircraft_list)
            airspace_bounds = f"lat({self.min_lat:.4f},{self.max_lat:.4f}), lon({self.min_lon:.4f},{self.max_lon:.4f})"
            
            # If we're tracking around an airport, include that information
            airport_info = ""
            if self.airport_info:
                # Use the correct key - the airport code could be stored as 'icao' or 'iata'
                airport_code = self.airport_info.get('icao', self.airport_info.get('iata', ''))
                airport_info = f"\nAirspace is around {self.airport_info['name']} ({airport_code}) airport."
                
            prompt = f"""Please provide a concise summary (5-6 lines) of the current airspace within bounds {airspace_bounds}{airport_info}
            
There are {total_aircraft} aircraft currently being tracked.

{aircraft_data}

In your summary:
1. Highlight any potentially critical aircraft (unusual altitude, speed, or vertical rate)
2. Categorize general traffic patterns (ascending, descending, cruising)
3. Note any clusters or congested areas
4. Mention aircraft that appear to be landing or taking off

Keep your response very concise, plain text only, and under 500 characters.
"""
            
            self.logger.debug(f"Sending airspace summary LLM request")
            
            # Prepare the API request
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }
            
            # Make the API request
            response = requests.post(url, json=payload, timeout=15)  # Increased timeout for larger prompt
            response.raise_for_status()
            
            # Parse the response
            self.logger.debug(f"Received LLM response for airspace summary")
            
            # Log the prompt to the summaries file, not to the main log
            self.logger.log_aircraft_summary("PROMPT", prompt)
            
            try:
                result = response.json()
                
                # Extract the generated text
                if "response" in result:
                    summary = result["response"].strip()
                    
                    # Save the summary, using "AIRSPACE" as a special callsign
                    with self.summary_lock:
                        self.aircraft_summary = summary
                        self.summary_callsign = "AIRSPACE"
                    
                    # Log the summary to dedicated file instead of main log
                    self.logger.log_aircraft_summary("AIRSPACE", summary)
                    
                    # Only log minimal info to main log
                    self.logger.debug(f"Processed airspace summary: {len(summary)} chars")
                    return summary
                else:
                    error_msg = "LLM response missing 'response' field"
                    self.logger.warning(f"{error_msg}: {result}")
                    return f"{error_msg}. Please try again."
            except json.JSONDecodeError:
                error_msg = "Failed to parse LLM response JSON"
                self.logger.error(f"{error_msg}: {response.text[:100]}...")
                return f"{error_msg}. Please try again."
                    
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making LLM request: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"Unexpected error getting airspace summary: {str(e)}\n{error_details}")
            return f"Error analyzing airspace: {str(e)}"
        finally:
            self.is_fetching_summary = False

    def fetch_summary_async(self, state=None):
        """Start a thread to fetch the airspace summary asynchronously"""
        if not self.use_llm or self.is_fetching_summary:
            return
            
        # Start a new thread to fetch the summary
        self.summary_thread = threading.Thread(
            target=self.get_aircraft_summary,
            args=(state,)
        )
        self.summary_thread.daemon = True
        self.summary_thread.start()

    def draw_llm_summary(self, start_y, max_x):
        """Draw the LLM summary in its own dedicated area"""
        if not self.use_llm:
            return
            
        # Get screen dimensions
        max_y, _ = self.stdscr.getmaxyx()
        
        # Draw section header with border
        section_title = "LLM Airspace Summary"
        separator = "=" * (max_x-1)
        
        # Draw section header and separator
        self.safe_addstr(start_y, 0, section_title, curses.A_BOLD)
        self.safe_addstr(start_y+1, 0, separator)
        
        # Get the summary content or status
        with self.lock:
            display_data = self.data if self.data else self.previous_valid_data
            if not display_data or 'states' not in display_data or not display_data['states']:
                self.safe_addstr(start_y+2, 2, "No aircraft data available for summary")
                return
                
            # Get selected aircraft for context (not used for summary generation anymore)
            states = display_data['states']
            if not states:
                self.safe_addstr(start_y+2, 2, "No aircraft in current airspace")
                return
                
            selected = None
            if 0 <= self.selected_idx < len(states):
                selected = states[self.selected_idx]
        
        # Display the summary or status message
        with self.summary_lock:
            summary_message = None
            
            if self.is_fetching_summary:
                summary_message = "Generating airspace summary..."
            elif self.aircraft_summary:
                summary_message = self.aircraft_summary
            else:
                summary_message = "No airspace summary available. Press 'r' to refresh."
        
        if summary_message:
            # Split the summary into multiple lines if needed
            max_line_width = max_x - 4  # Leave some margin
            summary_lines = []
            
            # Simple word wrapping
            words = summary_message.split()
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_line_width:
                    current_line += (" " + word) if current_line else word
                else:
                    summary_lines.append(current_line)
                    current_line = word
            
            if current_line:
                summary_lines.append(current_line)
            
            # Display each line of the summary, up to available space
            max_lines = 4  # Limit to 4 lines of summary text
            for i, line in enumerate(summary_lines[:max_lines]):
                if start_y + 2 + i < max_y:
                    self.safe_addstr(start_y + 2 + i, 2, line)
            
            # If we had to truncate the summary, show an indicator
            if len(summary_lines) > max_lines and start_y + 2 + max_lines < max_y:
                self.safe_addstr(start_y + 2 + max_lines, 2, "...")

    def draw_info_panel(self, start_y, max_x):
        """Draw the information panel for the selected aircraft"""
        with self.lock:
            display_data = self.data if self.data else self.previous_valid_data
            
            if not display_data or 'states' not in display_data or not display_data['states']:
                message = "Fetching flight data..." if self.refreshing_data else "No flight data available"
                self.safe_addstr(start_y, 0, message)
                return
                
            states = display_data['states']
            if not states or self.selected_idx < 0 or self.selected_idx >= len(states):
                return
                
            selected = states[self.selected_idx]
            
            # Get the color for the selected aircraft
            color_attr = self.get_aircraft_color(selected)
            
            # Start fetching airspace summary if LLM is enabled (passing selected for context)
            if self.use_llm:
                # Only request a new summary if we don't have one already or if it's been a while
                current_time = time.time()
                
                # Initialize last_summary_time if it doesn't exist
                if not hasattr(self, 'last_summary_time'):
                    self.last_summary_time = 0
                    
                summary_age = current_time - self.last_summary_time
                
                # Request a new summary if we don't have one or if it's older than llm_refresh_rate
                if not self.aircraft_summary or summary_age > self.llm_refresh_rate:
                    if not self.is_fetching_summary:
                        self.fetch_summary_async(selected)
                        # Update the last summary time
                        self.last_summary_time = current_time
        
        # Get screen dimensions
        max_y, max_width = self.stdscr.getmaxyx()
        
        # Display flight information with color
        # Display the aircraft details header
        headers = ["ICAO", "Callsign", "Country", "Position", "Altitude", "Speed", "Heading", "V.Rate"]
        header_str = " | ".join(headers)
        
        # Add aircraft position indicator after the headers
        aircraft_position = f"  Aircraft {self.selected_idx+1}/{len(states)}"
        
        # Combine headers and position indicator
        combined_header = header_str + aircraft_position
        self.safe_addstr(start_y, 0, combined_header, curses.A_BOLD)
        
        # Safely draw horizontal line
        separator = "-" * (max_x-1)
        self.safe_addstr(start_y+1, 0, separator)
        
        # Extract and format data
        icao24 = selected[0] or "N/A"
        callsign = selected[1].strip() if selected[1] else "N/A"
        country = selected[2] or "N/A"
        
        lat = selected[6]
        lon = selected[5]
        position = f"{lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else "N/A"
        
        baro_alt = selected[7]
        altitude = f"{int(baro_alt)}m" if baro_alt is not None else "N/A"
        
        velocity = selected[9]
        speed = f"{int(velocity)}m/s" if velocity is not None else "N/A"
        
        true_track = selected[10]
        heading = f"{int(true_track)}°" if true_track is not None else "N/A"
        
        vert_rate = selected[11]
        v_rate = f"{int(vert_rate)}m/s" if vert_rate is not None else "N/A"
        
        values = [icao24, callsign, country, position, altitude, speed, heading, v_rate]
        values_str = " | ".join(values)
        self.safe_addstr(start_y+2, 0, values_str, color_attr)
        
        # Additional data
        category_mapping = {
            0: "No info", 1: "No ADS-B info", 2: "Light", 3: "Small", 4: "Large",
            5: "High Vortex Large", 6: "Heavy", 7: "High Performance", 8: "Rotorcraft",
            9: "Glider", 10: "Lighter-than-air", 11: "Parachutist", 12: "Ultralight",
            13: "Reserved", 14: "UAV", 15: "Space Vehicle", 16: "Emergency Vehicle",
            17: "Service Vehicle", 18: "Point Obstacle", 19: "Cluster Obstacle", 20: "Line Obstacle"
        }
        
        position_source_mapping = {
            0: "ADS-B", 1: "ASTERIX", 2: "MLAT", 3: "FLARM"
        }
        
        category = selected[17] if len(selected) > 17 else None
        category_str = category_mapping.get(category, "Unknown") if category is not None else "Unknown"
        
        position_source = selected[16] if len(selected) > 16 else None
        position_source_str = position_source_mapping.get(position_source, "Unknown") if position_source is not None else "Unknown"
        
        on_ground = selected[8]
        on_ground_str = "Yes" if on_ground else "No" if on_ground is not None else "Unknown"
        
        squawk = selected[14] if len(selected) > 14 else None
        squawk_str = squawk if squawk else "N/A"
        
        geo_alt = selected[13] if len(selected) > 13 else None
        geo_alt_str = f"{int(geo_alt)}m" if geo_alt is not None else "N/A"
        
        time_position = selected[3]
        time_position_str = datetime.fromtimestamp(time_position).strftime('%H:%M:%S') if time_position else "N/A"
        
        last_contact = selected[4]
        last_contact_str = datetime.fromtimestamp(last_contact).strftime('%H:%M:%S') if last_contact else "N/A"
        
        additional_info = [
            f"Category: {category_str}",
            f"Position Source: {position_source_str}",
            f"On Ground: {on_ground_str}",
            f"Squawk: {squawk_str}",
            f"Geometric Alt: {geo_alt_str}",
            f"Last Position: {time_position_str}",
            f"Last Contact: {last_contact_str}"
        ]
        
        # Display additional info in two columns, safely
        col_width = max_width // 2
        for i, info in enumerate(additional_info):
            if start_y + 4 + (i % 3) < max_y:  # Make sure we're not writing below screen
                row = start_y + 4 + (i % 3)
                col = (i // 3) * col_width
                self.safe_addstr(row, col, info)
    
    def adjust_bounds(self, zoom_factor):
        """Adjust the bounds to zoom in or out"""
        # Calculate the center point
        center_lat = (self.min_lat + self.max_lat) / 2
        center_lon = (self.min_lon + self.max_lon) / 2
        
        # Calculate current width and height
        height = self.max_lat - self.min_lat
        width = self.max_lon - self.min_lon
        
        # Apply zoom factor
        new_height = height * zoom_factor
        new_width = width * zoom_factor
        
        # Calculate new bounds
        self.min_lat = center_lat - new_height / 2
        self.max_lat = center_lat + new_height / 2
        self.min_lon = center_lon - new_width / 2
        self.max_lon = center_lon + new_width / 2
        
        # Log the zoom action
        zoom_type = "in" if zoom_factor < 1 else "out"
        self.logger.info(f"Zoomed {zoom_type}: new bounds lat({self.min_lat:.4f},{self.max_lat:.4f}), lon({self.min_lon:.4f},{self.max_lon:.4f})")
        
        # Force refresh data
        self.fetch_data()
    
    def run(self):
        """Main loop for the application"""
        # Set up colors if terminal supports them
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)    # High altitude
            curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Medium altitude
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Low altitude
            curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)     # On ground
            curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # High speed
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)   # Default
        
        # Hide cursor
        curses.curs_set(0)
        
        # Don't wait for input when calling getch
        self.stdscr.nodelay(True)
        
        # Start the update thread
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        self.logger.info("Update thread started")
        
        # Main loop
        self.running = True
        # Add summary age tracking 
        self.last_summary_time = 0
        try:
            while self.running:
                try:
                    self.draw_map()
                    
                    # Check for key presses
                    key = self.stdscr.getch()
                    
                    if key == ord('q'):
                        self.running = False
                        # Explicitly set flag to stop thread
                        self.logger.info("User requested exit (q)")
                        # Important: break out of the loop to ensure proper cleanup
                        break
                    elif key == ord('p'):
                        # Toggle pause state
                        self.paused = not self.paused
                        if self.paused:
                            self.logger.info("User paused data updates")
                            self.status_message = "Data updates paused. Press 'p' to resume."
                        else:
                            self.logger.info("User resumed data updates")
                            self.status_message = "Data updates resumed."
                    elif key == ord('r'):
                        # Manual refresh
                        self.logger.info("Manual refresh requested (r)")
                        # Force fetch even if paused
                        self.fetch_data()
                    elif key == ord('+') or key == ord('='):
                        # Zoom in (decrease the area)
                        self.adjust_bounds(0.8)
                    elif key == ord('-'):
                        # Zoom out (increase the area)
                        self.adjust_bounds(1.2)
                    elif key == curses.KEY_UP:
                        with self.lock:
                            if self.data and 'states' in self.data and self.data['states']:
                                self.selected_idx = max(0, self.selected_idx - 1)
                                # Update selected callsign
                                states = self.data['states']
                                if states[self.selected_idx][1]:
                                    self.selected_callsign = states[self.selected_idx][1].strip()
                                else:
                                    self.selected_callsign = None
                                self.logger.debug(f"Selected aircraft index: {self.selected_idx}, callsign: {self.selected_callsign}")
                    elif key == curses.KEY_DOWN:
                        with self.lock:
                            if self.data and 'states' in self.data and self.data['states']:
                                self.selected_idx = min(len(self.data['states']) - 1, self.selected_idx + 1)
                                # Update selected callsign
                                states = self.data['states']
                                if states[self.selected_idx][1]:
                                    self.selected_callsign = states[self.selected_idx][1].strip()
                                else:
                                    self.selected_callsign = None
                                self.logger.debug(f"Selected aircraft index: {self.selected_idx}, callsign: {self.selected_callsign}")
                    elif key == curses.KEY_RESIZE:
                        # Handle terminal resize
                        self.stdscr.clear()
                        self.logger.info("Terminal resized")
                except Exception as e:
                    self.status_message = f"Error: {str(e)}"
                    self.logger.error(f"Error in main loop: {str(e)}")
                
                # Small delay to prevent high CPU usage
                time.sleep(0.1)
            
            # Clean up before exit
            curses.endwin()  # Restore terminal to normal state
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt detected, shutting down")
            curses.endwin()  # Restore terminal to normal state
        finally:
            # Ensure terminal is restored
            curses.endwin()  # Restore terminal to normal state
            self.logger.info("Main loop exited")

    def get_aircraft_color(self, state):
        """Determine the appropriate color for an aircraft based on its state"""
        on_ground = state[8]  # on_ground at index 8
        altitude = state[7]   # barometric altitude at index 7
        velocity = state[9]   # velocity at index 9
        
        if on_ground:
            return curses.color_pair(4)  # Red for aircraft on ground
        
        # Check if we have valid altitude and velocity
        if altitude is not None:
            if velocity is not None and velocity > 250:  # High speed aircraft (>250 m/s, ~900 km/h)
                return curses.color_pair(5)  # Magenta for high-speed aircraft
            
            # Color by altitude
            if altitude > 10000:  # High altitude (>10,000 meters, ~33,000 feet)
                return curses.color_pair(1)  # Cyan for high altitude
            elif altitude > 5000:  # Medium altitude (5,000-10,000 meters, ~16,500-33,000 feet)
                return curses.color_pair(2)  # Green for medium altitude
            else:  # Low altitude (<5,000 meters, ~16,500 feet)
                return curses.color_pair(3)  # Yellow for low altitude
        
        # Default color for aircraft with incomplete data
        return curses.color_pair(6)  # White for default
    
    def draw_legend(self, start_y, max_x):
        """Draw a legend explaining the color coding of aircraft"""
        # Define the legend items with their corresponding colors
        legend_items = [
            (curses.color_pair(1), "Cyan", "High altitude (>10,000m)"),
            (curses.color_pair(2), "Green", "Medium altitude (5,000-10,000m)"),
            (curses.color_pair(3), "Yellow", "Low altitude (<5,000m)"),
            (curses.color_pair(4), "Red", "On ground"),
            (curses.color_pair(5), "Magenta", "High speed (>250 m/s)"),
            (curses.color_pair(6), "White", "Unknown/Default")
        ]
        
        # Draw section header
        section_title = "Legend"
        separator = "-" * (max_x-1)
        
        # Draw section header and separator
        self.safe_addstr(start_y, 0, section_title, curses.A_BOLD)
        self.safe_addstr(start_y+1, 0, separator)
        
        # Determine layout (2 columns if enough space)
        cols = 2 if max_x >= 80 else 1
        col_width = max_x // cols - 4
        
        # Draw legend items
        for i, (color, name, desc) in enumerate(legend_items):
            row = i % (len(legend_items) // cols + (1 if len(legend_items) % cols else 0))
            col = i // (len(legend_items) // cols + (1 if len(legend_items) % cols else 0))
            
            # Add 2 to start_y to account for the title and separator
            y_pos = start_y + 2 + row
            x_pos = 2 + col * col_width
            
            # Draw colored symbol
            if x_pos < max_x - 2:
                try:
                    self.stdscr.addch(y_pos, x_pos, '•', color)
                    # Draw description
                    self.safe_addstr(y_pos, x_pos + 2, f"{name}: {desc}", color)
                except curses.error:
                    # Catch any errors from drawing outside the screen
                    pass

def main(stdscr, args):
    tracker = None
    try:
        tracker = FlightTracker(stdscr, args)
        tracker.run()
    except Exception as e:
        # Ensure the terminal is restored in case of unhandled exceptions
        curses.endwin()
        print(f"Error: {str(e)}")
    finally:
        # Make sure to clean up properly
        if tracker:
            tracker.running = False
            if tracker.update_thread:
                tracker.update_thread.join(timeout=1.0)
            # Close the logger
            if hasattr(tracker, 'logger'):
                tracker.logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time Terminal-based Flight Tracker')
    
    # Airport-based tracking
    parser.add_argument('--airport', type=str, help='Airport code (IATA or ICAO) to track flights around')
    parser.add_argument('--radius', type=float, default=20.0, help='Radius around airport in kilometers (default: 20)')
    parser.add_argument('--list-airports', action='store_true', help='List available airports and exit')
    
    # Geographic bounds (used if no airport is specified)
    parser.add_argument('--min-lat', type=float, default=45.8389, help='Minimum latitude')
    parser.add_argument('--max-lat', type=float, default=47.8229, help='Maximum latitude')
    parser.add_argument('--min-lon', type=float, default=5.9962, help='Minimum longitude')
    parser.add_argument('--max-lon', type=float, default=10.5226, help='Maximum longitude')
    
    # API authentication
    parser.add_argument('--username', type=str, help='OpenSky Network username')
    parser.add_argument('--password', type=str, help='OpenSky Network password')
    
    # Refresh rate
    parser.add_argument('--refresh', type=float, default=10.0, help='Data refresh rate in seconds (default: 10.0)')
    
    # Logging options
    parser.add_argument('--log-file', type=str, help='Path to log file (default: auto-generated in script directory)')
    parser.add_argument('--enable-logging', action='store_true', default=True, help='Enable logging (default: True)')
    parser.add_argument('--disable-logging', action='store_false', dest='enable_logging', help='Disable logging')
    
    # LLM options
    parser.add_argument('--use-llm', action='store_true', help='Enable LLM integration for aircraft summaries')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='URL for Ollama API (default: http://localhost:11434)')
    parser.add_argument('--ollama-model', type=str, default='gemma3:1b', help='Model to use for Ollama (default: gemma3:1b)')
    parser.add_argument('--llm-refresh', type=float, default=20.0, help='How often to refresh LLM summaries in seconds (default: 20.0)')
    
    args = parser.parse_args()
    
    # Handle --list-airports request
    if args.list_airports:
        print(list_airports())
        sys.exit(0)
    
    # If an airport was specified, validate it
    if args.airport:
        airport_code = args.airport.upper()
        if not get_airport_info(airport_code):
            print(f"Error: Airport code '{args.airport}' not found. Use --list-airports to see available airports.")
            sys.exit(1)
    
    # Initialize curses
    curses.wrapper(main, args)