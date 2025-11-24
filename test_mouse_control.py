#!/usr/bin/env python3
"""
Test mouse control methods for camera movement in The Dark Mod
"""

import subprocess
import time


def get_window_id(window_name):
    """Get the window ID by name"""
    try:
        cmd = ["xdotool", "search", "--name", window_name]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        window_ids = output.split('\n')
        return window_ids[0] if window_ids else None
    except Exception as e:
        print(f"Error finding window: {e}")
        return None


def test_mousemove_relative(window_id, x, y):
    """Test 1: xdotool mousemove_relative"""
    print(f"\n🔧 Test 1: xdotool mousemove_relative (x={x}, y={y})")
    try:
        # Focus window first
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.1)
        
        # Move mouse relatively
        subprocess.run(["xdotool", "mousemove_relative", "--", str(x), str(y)],
                      check=True, stderr=subprocess.DEVNULL)
        print(f"✅ Success - Moved {x}, {y}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_mousemove_relative_with_sync(window_id, x, y):
    """Test 2: xdotool mousemove_relative with --sync"""
    print(f"\n🔧 Test 2: xdotool mousemove_relative --sync (x={x}, y={y})")
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.1)
        
        # Move mouse with sync
        subprocess.run(["xdotool", "mousemove_relative", "--sync", "--", str(x), str(y)],
                      check=True, stderr=subprocess.DEVNULL)
        print(f"✅ Success - Moved {x}, {y}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_mousemove_to_center_then_relative(window_id, x, y):
    """Test 3: Move to window center, then relative"""
    print(f"\n🔧 Test 3: Move to center, then relative (x={x}, y={y})")
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.1)
        
        # Get window geometry
        cmd_geo = ["xdotool", "getwindowgeometry", window_id]
        output = subprocess.check_output(cmd_geo, stderr=subprocess.DEVNULL).decode()
        
        # Parse position and size
        for line in output.split('\n'):
            if "Position:" in line:
                pos_str = line.split("Position:")[1].split("(")[0].strip()
                win_x, win_y = map(int, pos_str.split(","))
            if "Geometry:" in line:
                geo_str = line.split("Geometry:")[1].strip()
                win_w, win_h = map(int, geo_str.split("x"))
        
        # Calculate center
        center_x = win_x + win_w // 2
        center_y = win_y + win_h // 2
        
        # Move to center
        subprocess.run(["xdotool", "mousemove", str(center_x), str(center_y)],
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.1)
        
        # Now move relatively
        subprocess.run(["xdotool", "mousemove_relative", "--", str(x), str(y)],
                      check=True, stderr=subprocess.DEVNULL)
        print(f"✅ Success - Centered then moved {x}, {y}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_click_and_drag(window_id, x, y):
    """Test 4: Click and drag (simulates mouse look in FPS games)"""
    print(f"\n🔧 Test 4: Click and drag (x={x}, y={y})")
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.1)
        
        # Get window center
        cmd_geo = ["xdotool", "getwindowgeometry", window_id]
        output = subprocess.check_output(cmd_geo, stderr=subprocess.DEVNULL).decode()
        
        for line in output.split('\n'):
            if "Position:" in line:
                pos_str = line.split("Position:")[1].split("(")[0].strip()
                win_x, win_y = map(int, pos_str.split(","))
            if "Geometry:" in line:
                geo_str = line.split("Geometry:")[1].strip()
                win_w, win_h = map(int, geo_str.split("x"))
        
        center_x = win_x + win_w // 2
        center_y = win_y + win_h // 2
        
        # Move to center and press mouse button
        subprocess.run(["xdotool", "mousemove", str(center_x), str(center_y)],
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.05)
        
        subprocess.run(["xdotool", "mousedown", "1"],
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.05)
        
        # Drag
        subprocess.run(["xdotool", "mousemove_relative", "--", str(x), str(y)],
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.05)
        
        subprocess.run(["xdotool", "mouseup", "1"],
                      check=True, stderr=subprocess.DEVNULL)
        
        print(f"✅ Success - Clicked and dragged {x}, {y}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_repeated_small_movements(window_id, x, y, steps=10):
    """Test 5: Multiple small movements instead of one large one"""
    print(f"\n🔧 Test 5: {steps} small movements (total x={x}, y={y})")
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.1)
        
        step_x = x // steps
        step_y = y // steps
        
        for i in range(steps):
            subprocess.run(["xdotool", "mousemove_relative", "--", str(step_x), str(step_y)],
                          check=True, stderr=subprocess.DEVNULL)
            time.sleep(0.02)  # Small delay between steps
        
        print(f"✅ Success - Moved in {steps} steps")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    game_name = "The Dark Mod"
    
    print("=" * 80)
    print("🖱️  MOUSE CONTROL TEST FOR CAMERA MOVEMENT")
    print("=" * 80)
    
    window_id = get_window_id(game_name)
    if not window_id:
        print(f"❌ Could not find window '{game_name}'")
        return
    
    print(f"✅ Found window ID: {window_id}\n")
    print("=" * 80)
    print("Testing different mouse control methods...")
    print("Watch the game - camera should rotate")
    print("=" * 80)
    
    test_x = 300  # Move right
    test_y = 0
    
    try:
        # Test 1
        input("\nPress ENTER to test Method 1 (mousemove_relative)...")
        test_mousemove_relative(window_id, test_x, test_y)
        time.sleep(2)
        
        # Test 2
        input("\nPress ENTER to test Method 2 (mousemove_relative --sync)...")
        test_mousemove_relative_with_sync(window_id, test_x, test_y)
        time.sleep(2)
        
        # Test 3
        input("\nPress ENTER to test Method 3 (center then relative)...")
        test_mousemove_to_center_then_relative(window_id, test_x, test_y)
        time.sleep(2)
        
        # Test 4
        input("\nPress ENTER to test Method 4 (click and drag)...")
        test_click_and_drag(window_id, test_x, test_y)
        time.sleep(2)
        
        # Test 5
        input("\nPress ENTER to test Method 5 (small repeated movements)...")
        test_repeated_small_movements(window_id, test_x, test_y, steps=10)
        time.sleep(2)
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETE")
        print("=" * 80)
        print("\n📊 NOTES:")
        print("   - The Dark Mod may use mouse capture/lock")
        print("   - FPS games often ignore xdotool mouse movement")
        print("   - Alternative: Use keyboard for camera (arrow keys, or mouse keys)")
        print("   - Check if game has 'mouse look' toggle or free cursor mode")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Tests stopped")


if __name__ == "__main__":
    main()
