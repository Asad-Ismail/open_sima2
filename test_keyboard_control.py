#!/usr/bin/env python3
"""
Test alternative methods without window activation overhead
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


def method_a_focus_once(window_id, key, duration=0.3):
    """Focus window ONCE, then send keys without refocusing"""
    print(f"\n🔧 Method A: Focus once, then keydown/keyup (no refocus)")
    try:
        start = time.time()
        # Focus window ONCE
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], 
                      check=True, stderr=subprocess.DEVNULL)
        print("   Window focused, now sending keys...")
        time.sleep(0.1)
        
        # Send keys without refocusing
        subprocess.run(["xdotool", "keydown", key], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(duration)
        subprocess.run(["xdotool", "keyup", key], 
                      check=True, stderr=subprocess.DEVNULL)
        
        latency = (time.time() - start) * 1000
        print(f"✅ Success - Latency: {latency:.1f}ms")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def method_b_xte(window_id, key, duration=0.3):
    """Use xte (X Test Events) instead of xdotool"""
    print(f"\n🔧 Method B: xte (X Test Events)")
    try:
        start = time.time()
        # Check if xte is available
        subprocess.run(["which", "xte"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Focus window once
        subprocess.run(["xdotool", "windowactivate", window_id], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.05)
        
        # Send key using xte (faster)
        subprocess.run(["xte", f"'key {key}'"], 
                      shell=True, check=True, stderr=subprocess.DEVNULL)
        time.sleep(duration)
        
        latency = (time.time() - start) * 1000
        print(f"✅ Success - Latency: {latency:.1f}ms")
        return True
    except subprocess.CalledProcessError:
        print("❌ xte not installed. Install with: sudo apt-get install xautomation")
        return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def method_c_xdotool_without_sync(window_id, key, duration=0.3):
    """Use xdotool windowactivate WITHOUT --sync flag"""
    print(f"\n🔧 Method C: windowactivate without --sync (async)")
    try:
        start = time.time()
        # Focus window without waiting
        subprocess.run(["xdotool", "windowactivate", window_id], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.05)  # Small delay
        
        subprocess.run(["xdotool", "keydown", key], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(duration)
        subprocess.run(["xdotool", "keyup", key], 
                      check=True, stderr=subprocess.DEVNULL)
        
        latency = (time.time() - start) * 1000
        print(f"✅ Success - Latency: {latency:.1f}ms")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def method_d_windowfocus(window_id, key, duration=0.3):
    """Use windowfocus instead of windowactivate"""
    print(f"\n🔧 Method D: windowfocus (lighter than windowactivate)")
    try:
        start = time.time()
        subprocess.run(["xdotool", "windowfocus", window_id], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(0.05)
        
        subprocess.run(["xdotool", "keydown", key], 
                      check=True, stderr=subprocess.DEVNULL)
        time.sleep(duration)
        subprocess.run(["xdotool", "keyup", key], 
                      check=True, stderr=subprocess.DEVNULL)
        
        latency = (time.time() - start) * 1000
        print(f"✅ Success - Latency: {latency:.1f}ms")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    game_name = "The Dark Mod"
    
    print("=" * 80)
    print("🚀 FAST CONTROL METHODS TEST")
    print("=" * 80)
    
    window_id = get_window_id(game_name)
    if not window_id:
        print(f"❌ Could not find window '{game_name}'")
        return
    
    print(f"✅ Found window ID: {window_id}\n")
    
    test_key = "w"
    test_duration = 0.3
    
    print("=" * 80)
    print("Testing alternative methods to reduce latency...")
    print("Watch the game window - character should move")
    print("=" * 80)
    
    try:
        # Method A: Focus once, then just send keys
        input("\nPress ENTER to test Method A (Focus once)...")
        method_a_focus_once(window_id, test_key, test_duration)
        time.sleep(1)
        
        # Test multiple actions without refocusing
        print("\n   Testing multiple actions without refocusing:")
        for i in range(3):
            start = time.time()
            subprocess.run(["xdotool", "keydown", test_key], 
                          check=True, stderr=subprocess.DEVNULL)
            time.sleep(test_duration)
            subprocess.run(["xdotool", "keyup", test_key], 
                          check=True, stderr=subprocess.DEVNULL)
            latency = (time.time() - start) * 1000
            print(f"   Action {i+1}: {latency:.1f}ms")
            time.sleep(0.5)
        
        # Method B: xte
        input("\nPress ENTER to test Method B (xte)...")
        method_b_xte(window_id, test_key, test_duration)
        time.sleep(1)
        
        # Method C: async windowactivate
        input("\nPress ENTER to test Method C (async windowactivate)...")
        method_c_xdotool_without_sync(window_id, test_key, test_duration)
        time.sleep(1)
        
        # Method D: windowfocus
        input("\nPress ENTER to test Method D (windowfocus)...")
        method_d_windowfocus(window_id, test_key, test_duration)
        
        print("\n" + "=" * 80)
        print("✅ TESTS COMPLETE")
        print("=" * 80)
        print("\n📊 RECOMMENDATION:")
        print("   Best: Method A (Focus once at startup, no refocus)")
        print("   - Latency: ~300-350ms (just the key hold time)")
        print("   - No window activation overhead per action")
        print("   - Requires game window to stay focused")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Tests stopped")


if __name__ == "__main__":
    main()
