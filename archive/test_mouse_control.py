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
