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


def execute_action(window_id, action_key, duration=0.2):
    """Execute a keyboard action directly to the game window using xdotool"""
    if action_key.lower() == 'none':
        return
    
    try:
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
        
        key = key_map.get(action_key.lower())
        if key is None:
            print(f"  ⚠ Unknown key: {action_key}")
            return
        
        # Send keydown
        subprocess.run(["xdotool", "key", "--window", window_id, key], check=True)
        print(f"  → Executed: {action_key}")
        
    except Exception as e:
        print(f"  ⚠ Failed to execute {action_key}: {e}")


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
    print("⏳ Starting agent in 5 seconds... (Press Ctrl+C to stop)")
    time.sleep(5)
    
    frame_count = 0
    decision_interval = 2  # Make decisions every N frames
    last_action = None
    
    with mss.mss() as sct:
        try:
            while True:
                # Get dynamic window position
                region = get_window_geometry(game_name)
                
                if not region:
                    print("⚠ Game window not found! Retrying in 2 seconds...")
                    time.sleep(2)
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
                            instruction="Analyze the current game state and decide the next move.",
                            max_new_tokens=256,
                            temperature=0.7
                        )
                        
                        print(f"\n💭 Reasoning: {action.reasoning[:150]}...")
                        print(f"👁 Visibility: {action.visibility_status}")
                        print(f"⚠ Threats: {action.detected_threats}")
                        print(f"🎯 Action: {action.recommended_action}")
                        print(f"📝 Explanation: {action.action_explanation}")
                        
                        # Execute the action
                        if action.recommended_action != last_action:
                            execute_action(window_id, action.recommended_action)
                            last_action = action.recommended_action
                    
                    # Display the game view (optional)
                    display_img = cv2.resize(img_np, (800, 600))
                    cv2.putText(display_img, f"Frame: {frame_count}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if frame_count % decision_interval == 0 and last_action:
                        cv2.putText(display_img, f"Action: {last_action}", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("AI Agent View", display_img)
                    
                    frame_count += 1
                    
                except Exception as e:
                    print(f"⚠ Capture/Processing failed: {e}")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n👋 Quitting agent...")
                    break
                
                # Small delay to not overwhelm the system
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Agent stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            print("✅ Cleanup complete")


if __name__ == "__main__":
    main()