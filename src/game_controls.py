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


def execute_action_sequence(action_sequence: list, hold_duration: float = 0.1):
    """
    Execute keyboard actions SIMULTANEOUSLY (like human input).
    Matches recording behavior where multiple keys can be pressed at once.
    
    Args:
        action_sequence: List of action keys (can include simultaneous keys)
        hold_duration: How long to hold keys (should match recording interval: 0.1s)
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
        'enter': 'Return',
        'up': 'Up',
        'down': 'Down',
        'left': 'Left',
        'right': 'Right'
    }
    
    # Filter out 'none' actions
    valid_keys = [k for k in action_sequence if k.lower() != 'none']
    
    if not valid_keys:
        return
    
    print(f"\n  🎮 Executing: {valid_keys}")
    
    # Press all keys down simultaneously
    pressed_keys = []
    for action_key in valid_keys:
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
            pressed_keys.append(key)
            
        except Exception as e:
            print(f"    ⚠ Failed to press {action_key}: {e}")
    
    # Hold for duration (matches recording frame interval)
    time.sleep(hold_duration)
    
    # Release all keys
    for key in pressed_keys:
        try:
            subprocess.run(
                ["xdotool", "keyup", key],
                check=True,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"    ⚠ Failed to release {key}: {e}")
    
    print(f"    ✓ Held {len(pressed_keys)} keys for {hold_duration}s")


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
