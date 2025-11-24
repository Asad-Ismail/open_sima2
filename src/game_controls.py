"""
Game control utilities for keyboard and mouse input.
"""

import subprocess
import time


def execute_mouse_look(mouse_data: dict):
    """
    Execute mouse movement for camera control using xdotool.
    
    Args:
        mouse_data: Dictionary with 'x' and 'y' values for mouse movement
                   x: -100 to 100 (left/right), y: -50 to 50 (up/down)
    """
    x = mouse_data.get('x', 0)
    y = mouse_data.get('y', 0)
    
    if x == 0 and y == 0:
        return
    
    print(f"\n  🖱️  Mouse Look: x={x}, y={y}")
    
    try:
        scale_factor = 3
        scaled_x = int(x * scale_factor)
        scaled_y = int(y * scale_factor)
        
        subprocess.run(
            ["xdotool", "mousemove_relative", "--", str(scaled_x), str(scaled_y)],
            check=True,
            stderr=subprocess.DEVNULL
        )
        
        print(f"    ✓ Moved camera: ({scaled_x}, {scaled_y}) pixels")
        time.sleep(0.05)
        
    except Exception as e:
        print(f"    ⚠ Failed to execute mouse look: {e}")


def execute_action_sequence(action_sequence: list, duration: float = 0.3, pause_between: float = 0.2):
    """
    Execute a sequence of keyboard actions using xdotool.
    
    Args:
        action_sequence: List of action keys to execute
        duration: How long to hold each key (seconds)
        pause_between: Pause between actions (seconds)
    """
    if not action_sequence or (len(action_sequence) == 1 and action_sequence[0].lower() == 'none'):
        return
    
    key_map = {
        'w': 'w',
        's': 's',
        'a': 'a',
        'd': 'd',
        'c': 'c',
        'space': 'space',
        'enter': 'Return'
    }
    
    print(f"\n  🎮 Executing action sequence: {action_sequence}")
    
    for i, action_key in enumerate(action_sequence):
        if action_key.lower() == 'none':
            continue
            
        try:
            key = key_map.get(action_key.lower())
            if key is None:
                print(f"    ⚠ Unknown key: {action_key}")
                continue
            
            subprocess.run(
                ["xdotool", "keydown", key],
                check=True,
                stderr=subprocess.DEVNULL
            )
            time.sleep(duration)
            subprocess.run(
                ["xdotool", "keyup", key],
                check=True,
                stderr=subprocess.DEVNULL
            )
            print(f"    [{i+1}/{len(action_sequence)}] → {action_key}")
            
            if i < len(action_sequence) - 1:
                time.sleep(pause_between)
            
        except Exception as e:
            print(f"    ⚠ Failed to execute {action_key}: {e}")


def focus_window(window_id: str):
    """
    Focus a window by its ID.
    
    Args:
        window_id: Window ID to focus
    """
    subprocess.run(
        ["xdotool", "windowactivate", "--sync", window_id],
        check=True,
        stderr=subprocess.DEVNULL
    )
