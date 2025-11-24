"""
Screen capture utilities for game window management.
"""

import mss
import numpy as np
import subprocess


def get_window_id(window_name: str) -> str:
    """
    Get the window ID by name.
    
    Args:
        window_name: Fragment of window title to search for
        
    Returns:
        Window ID string or None if not found
    """
    try:
        cmd = ["xdotool", "search", "--name", window_name]
        window_id = subprocess.check_output(cmd).decode().strip().split('\n')[0]
        return window_id
    except Exception as e:
        print(f"Error finding window: {e}")
        return None


def get_window_geometry(window_name: str) -> dict:
    """
    Get the position and size of a window by name.
    
    Args:
        window_name: Fragment of window title to search for
        
    Returns:
        Dictionary with 'top', 'left', 'width', 'height' or None if not found
    """
    try:
        cmd_id = ["xdotool", "search", "--name", window_name]
        window_id = subprocess.check_output(cmd_id).decode().strip().split('\n')[0]
        
        cmd_geo = ["xdotool", "getwindowgeometry", window_id]
        output = subprocess.check_output(cmd_geo).decode()
        
        lines = output.split('\n')
        position_line = [l for l in lines if "Position:" in l][0]
        geometry_line = [l for l in lines if "Geometry:" in l][0]
        
        pos_str = position_line.split("Position:")[1].split("(")[0].strip()
        x, y = map(int, pos_str.split(","))
        geo_str = geometry_line.split("Geometry:")[1].strip()
        w, h = map(int, geo_str.split("x"))
        
        return {"top": y, "left": x, "width": w, "height": h}
    except Exception as e:
        print(f"Error getting window geometry: {e}")
        return None


def validate_region(region: dict, screen_width: int, screen_height: int) -> dict:
    """
    Validate and adjust region to screen bounds.
    
    Args:
        region: Dictionary with 'top', 'left', 'width', 'height'
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        
    Returns:
        Validated region dictionary
    """
    if region["left"] < 0:
        region["left"] = 0
    if region["top"] < 0:
        region["top"] = 0
    
    if (region["left"] + region["width"]) > screen_width:
        region["width"] = screen_width - region["left"]
    
    if (region["top"] + region["height"]) > screen_height:
        region["height"] = screen_height - region["top"]
    
    return region
