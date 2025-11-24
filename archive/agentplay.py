"""
Autonomous Dark Mod Agent - Combines Qwen Vision Model with Screen Capture
Integrates real-time game screen capture, vision model inference, and keyboard control
"""

import time
import mss
import numpy as np
import cv2
import subprocess
from PIL import Image
import sys

# Import our Qwen Vision Model
from qwen_vision_model import QwenVisionAgent


def get_window_id(window_name_fragment):
    """Get the window ID by name"""
    try:
        cmd_id = ["xdotool", "search", "--name", window_name_fragment]
        window_id = subprocess.check_output(cmd_id).decode().strip().split('\n')[0]
        return window_id
    except Exception as e:
        print(f"Error finding window: {e}")
        return None


def get_window_geometry(window_name_fragment):
    """Get the position and size of a window by name"""
    try:
        cmd_id = ["xdotool", "search", "--name", window_name_fragment]
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
        print(f"Error finding window: {e}")
        return None


def execute_mouse_look(mouse_data):
    """Execute mouse movement for camera control using xdotool mousemove_relative
    Uses Method 1: Simple relative movement (lowest latency, ~10-20ms)
    Compatible with keyboard controls - no window refocus needed
    
    Args:
        mouse_data: Dictionary with 'x' and 'y' values for mouse movement
                   x: -100 to 100 (left/right), y: -50 to 50 (up/down)
    """
    x = mouse_data.get('x', 0)
    y = mouse_data.get('y', 0)
    
    # Skip if no movement
    if x == 0 and y == 0:
        return
    
    print(f"\n  🖱️  Mouse Look: x={x}, y={y}")
    
    try:
        # Scale movements (multiply by factor for smoother control)
        # Positive x = right, Negative x = left
        # Positive y = down, Negative y = up
        scale_factor = 3  # Adjust sensitivity
        scaled_x = int(x * scale_factor)
        scaled_y = int(y * scale_factor)
        
        #  Simple mousemove_relative (lowest latency)
        # Window must already be focused (done at startup)
        subprocess.run(["xdotool", "mousemove_relative", "--", str(scaled_x), str(scaled_y)],
                      check=True, stderr=subprocess.DEVNULL)
        
        print(f"    ✓ Moved camera: ({scaled_x}, {scaled_y}) pixels")
        time.sleep(0.05)  # Minimal delay for game to register movement
        
    except Exception as e:
        print(f"    ⚠ Failed to execute mouse look: {e}")


def execute_action_sequence(action_sequence, duration=0.3, pause_between=0.2):
    """Execute a sequence of keyboard actions using xdotool keydown/keyup
    Window must already be focused (done once at startup)
    
    Args:
        action_sequence: List of action keys to execute
        duration: How long to hold each key (seconds)
        pause_between: Pause between actions (seconds)
    """
    if not action_sequence or (len(action_sequence) == 1 and action_sequence[0].lower() == 'none'):
        return
    
    # Map action to xdotool key
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
            
            # Send keydown/keyup (no refocus - window already focused)
            subprocess.run(["xdotool", "keydown", key], 
                          check=True, stderr=subprocess.DEVNULL)
            time.sleep(duration)
            subprocess.run(["xdotool", "keyup", key], 
                          check=True, stderr=subprocess.DEVNULL)
            print(f"    [{i+1}/{len(action_sequence)}] → {action_key}")
            
            # Pause between actions (except after last action)
            if i < len(action_sequence) - 1:
                time.sleep(pause_between)
            
        except Exception as e:
            print(f"    ⚠ Failed to execute {action_key}: {e}")


def get_user_goal():
    """Get new goal from user"""
    print("\n" + "="*80)
    print("💬 Enter new goal (or 'quit' to exit):")
    print("="*80)
    goal = input("🎯 Goal: ").strip()
    return goal


def main():
    print("=" * 80)
    print("🎮 AUTONOMOUS DARK MOD AGENT")
    print("=" * 80)
    print("\n🧠 Loading Qwen Vision Model...")
    
    # Initialize the Qwen Vision Agent
    agent = QwenVisionAgent()
    
    # Game window name
    game_name = "The Dark Mod"
    
    # Get game window ID
    window_id = get_window_id(game_name)
    if not window_id:
        print("❌ Error: Could not find game window!")
        print(f"   Make sure '{game_name}' is running.")
        return
    
    print("\n✅ Model loaded successfully!")
    print(f"\n🎯 Looking for game window: '{game_name}'")
    
    # Focus the game window ONCE at startup (avoids 15s overhead per action)
    print("\n🔧 Focusing game window...")
    subprocess.run(["xdotool", "windowactivate", "--sync", window_id], 
                  check=True, stderr=subprocess.DEVNULL)
    print("✅ Window focused! (Keep game window in focus for best performance)")
    print("⏳ Starting agent in 2 seconds... (Press Ctrl+C to stop)")
    time.sleep(2)
    
    frame_count = 0
    decision_interval = 3  # Make decisions every N frames
    goal_reached = False
    frames_not_facing = 0  # Track how long we haven't seen the target
    
    with mss.mss() as sct:
        try:
            while True:
                # If goal is reached, wait for user to provide new goal
                if goal_reached:
                    print("\n" + "="*80)
                    print("✅ GOAL REACHED! Agent paused.")
                    print("="*80)
                    new_goal = get_user_goal()
                    
                    if new_goal.lower() == 'quit':
                        print("\n👋 Exiting agent...")
                        break
                    
                    # Update agent's goal and restart
                    print(f"\n🎯 New goal set: {new_goal}")
                    agent.system_prompt = agent.system_prompt.split("Your goal is to")[0] + f"Your goal is to {new_goal}. Remember you are in 3D environment" + agent.system_prompt.split("and must use the provided controls")[1]
                    goal_reached = False
                    frame_count = 0
                    frames_not_facing = 0
                    print("⏳ Resuming in 2 seconds...")
                    time.sleep(2)
                    continue
                
                # Get dynamic window position
                region = get_window_geometry(game_name)
                
                if not region:
                    print("⚠ Game window not found! Retrying in 2 seconds...")
                    time.sleep(1)
                    continue
                
                # Validate and adjust region to screen bounds
                screen = sct.monitors[1]
                screen_w = screen["width"]
                screen_h = screen["height"]
                
                if region["left"] < 0:
                    region["left"] = 0
                if region["top"] < 0:
                    region["top"] = 0
                if (region["left"] + region["width"]) > screen_w:
                    region["width"] = screen_w - region["left"]
                if (region["top"] + region["height"]) > screen_h:
                    region["height"] = screen_h - region["top"]
                
                try:
                    # Capture the screen
                    sct_img = sct.grab(region)
                    img_np = np.array(sct_img)
                    
                    # Convert to PIL Image for model
                    img_pil = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    
                    # Make decisions at intervals
                    if frame_count % decision_interval == 0:
                        print(f"\n{'='*60}")
                        print(f"Frame: {frame_count}")
                        print("🤔 Analyzing scene...")
                        
                        # Get action from vision model
                        action = agent.analyze_frame(
                            img_pil,
                            instruction="Analyze the current game state and decide the next actions.",
                            max_new_tokens=300,
                            temperature=0.7
                        )
                        
                        print(f"\n💭 Reasoning: {action.reasoning[:150]}...")
                        print(f"👁 Visibility: {action.visibility_status}")
                        print(f"⚠ Threats: {action.detected_threats}")
                        print(f"🎯 Facing Target: {action.facing_target}")
                        print(f"🖱️  Mouse Look: {action.mouse_look}")
                        print(f"🎮 Actions: {action.action_sequence}")
                        print(f"📝 Explanation: {action.action_explanation}")
                        print(f"🏁 Goal Status: {action.goal_status}")
                        
                        # Check if goal is reached
                        if action.goal_reached:
                            goal_reached = True
                            print("\n✨ ✅ GOAL REACHED! ✅ ✨")
                            continue
                        
                        # Track if not facing target
                        if not action.facing_target:
                            frames_not_facing += 1
                            print(f"\n🔍 Target not visible (frame {frames_not_facing}) - AI searching...")
                            
                            # Warn AI if it's not searching actively
                            if action.mouse_look.get('x', 0) == 0 and action.mouse_look.get('y', 0) == 0:
                                print(f"   ⚠️  WARNING: AI not moving camera! (Should be searching)")
                            
                            # Execute AI's search movement
                            if action.mouse_look:
                                execute_mouse_look(action.mouse_look)
                        else:
                            # Target is visible, reset search tracking
                            frames_not_facing = 0
                            search_phase = 0
                            
                            # Execute agent's mouse look if any
                            if action.mouse_look:
                                execute_mouse_look(action.mouse_look)
                        
                        # Execute the action sequence
                        if action.action_sequence:
                            execute_action_sequence(action.action_sequence)
                    
                    # Display the game view (optional)
                    display_img = cv2.resize(img_np, (800, 600))
                    cv2.putText(display_img, f"Frame: {frame_count}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    #cv2.imshow("AI Agent View", display_img)
                    
                    frame_count += 1
                    
                except Exception as e:
                    print(f"⚠ Capture/Processing failed: {e}")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n👋 Quitting agent...")
                    break
        
        except KeyboardInterrupt:
            print("\n\n🛑 Agent stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            print("✅ Cleanup complete")


if __name__ == "__main__":
    main()