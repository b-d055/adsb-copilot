#!/usr/bin/env python3
"""
Logging module for the ADS-B Flight Tracker application.
Provides functionality to log application events to a file for debugging.
"""
import os
import sys
import time
from datetime import datetime
import threading
from typing import Optional, TextIO, Dict, Any

class Logger:
    """Logger class for ADS-B Flight Tracker"""
    
    def __init__(self, log_file: str = None, enabled: bool = True):
        """
        Initialize the logger.
        
        Args:
            log_file (str): Path to the log file. If None, a default name will be used.
            enabled (bool): Whether logging is enabled
        """
        self.enabled = enabled
        self.lock = threading.Lock()
        self.file_handle: Optional[TextIO] = None
        
        if not log_file:
            # Create a log file in the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(script_dir, f"adsb_tracker_{timestamp}.log")
        
        self.log_file = log_file
        
        if self.enabled:
            try:
                self.file_handle = open(self.log_file, 'a', encoding='utf-8')
                self.info(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.info(f"Log file: {self.log_file}")
            except Exception as e:
                self.enabled = False
                print(f"Error opening log file: {str(e)}", file=sys.stderr)
    
    def __del__(self):
        """Close the log file when the logger is destroyed"""
        try:
            self.close()
        except:
            # Suppress any errors during deletion
            pass
    
    def close(self):
        """Close the log file"""
        if self.file_handle:
            try:
                # Use a timeout when acquiring the lock to prevent deadlocks during shutdown
                if self.lock.acquire(timeout=1.0):
                    try:
                        self.info(f"Log closed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        self.file_handle.close()
                    finally:
                        self.lock.release()
                else:
                    # If we couldn't acquire the lock, still try to close the file
                    self.file_handle.close()
            except Exception:
                # Suppress errors during close as we're shutting down anyway
                pass
            finally:
                self.file_handle = None
    
    def _write(self, level: str, message: str):
        """
        Write a message to the log file.
        
        Args:
            level (str): Log level (INFO, WARNING, ERROR, DEBUG)
            message (str): Message to log
        """
        if not self.enabled or not self.file_handle:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        # Use a timeout when acquiring the lock to prevent deadlocks
        acquired = False
        try:
            acquired = self.lock.acquire(timeout=0.5)
            if acquired and self.file_handle:
                try:
                    self.file_handle.write(log_line)
                    self.file_handle.flush()  # Ensure it's written immediately
                except Exception:
                    # If we can't write to the log file, disable logging
                    self.enabled = False
        except Exception:
            # If anything goes wrong with lock acquisition, just continue
            pass
        finally:
            # Only release if we actually acquired the lock
            if acquired:
                self.lock.release()
    
    def info(self, message: str):
        """Log an informational message"""
        self._write("INFO", message)
    
    def warning(self, message: str):
        """Log a warning message"""
        self._write("WARNING", message)
    
    def error(self, message: str):
        """Log an error message"""
        self._write("ERROR", message)
    
    def debug(self, message: str):
        """Log a debug message"""
        self._write("DEBUG", message)
    
    def log_request(self, url: str, status_code: Optional[int] = None, 
                    duration: Optional[float] = None, error: Optional[str] = None):
        """
        Log an API request.
        
        Args:
            url (str): The URL that was requested
            status_code (int): HTTP status code
            duration (float): Request duration in seconds
            error (str): Error message if the request failed
        """
        if error:
            self.error(f"API Request Failed: {url} - Error: {error}")
        elif status_code:
            self.info(f"API Request: {url} - Status: {status_code} - Duration: {duration:.2f}s")
        else:
            self.info(f"API Request: {url}")
    
    def log_data_update(self, stats: Dict[str, Any]):
        """
        Log data update statistics.
        
        Args:
            stats (dict): Statistics about the data update
        """
        message = "Data Update: "
        message += ", ".join([f"{k}={v}" for k, v in stats.items()])
        self.info(message)

# Global logger instance
_logger = None

def get_logger(log_file: str = None, enabled: bool = True) -> Logger:
    """
    Get the global logger instance.
    
    Args:
        log_file (str): Path to the log file
        enabled (bool): Whether logging is enabled
        
    Returns:
        Logger: The global logger instance
    """
    global _logger
    if _logger is None:
        _logger = Logger(log_file, enabled)
    return _logger

if __name__ == "__main__":
    # Test the logger
    logger = get_logger()
    logger.info("This is a test log message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    logger.log_request("https://example.com/api", 200, 0.5)
    logger.log_request("https://example.com/api/error", error="Connection refused")
    logger.log_data_update({"total_aircraft": 10, "filtered": 2, "duration": 1.5})
    logger.close()
    print(f"Test log written to {logger.log_file}")
