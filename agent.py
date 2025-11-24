"""
Autonomous game playing agent that combines vision and control.
Integrates real-time game screen capture, vision model inference, and game controls.
"""

import time
import mss
import numpy as np
import cv2
from PIL import Image
import sys

from src.vision_agent import VisionAgent
from src.screen_capture import get_window_id, get_window_geometry, validate_region
from src.game_controls import execute_mouse_look, execute_action_sequence, focus_window


def get_user_goal() -> str:
    """Get new goal from user."""
    print("\n" + "="*80)
    print("💬 Enter new goal (or 'quit' to exit):")
    print("="*80)
    goal = input("🎯 Goal: ").strip()
    return goal


def main():
    print("=" * 80)
    print("🎮 AUTONOMOUS GAME AGENT")
    print("=" * 80)
    print("\n🧠 Loading Vision Model...")
    
    agent = VisionAgent()
    
    game_name = "The Dark Mod"
    
    window_id = get_window_id(game_name)
    if not window_id:
        print("❌ Error: Could not find game window!")
        print(f"   Make sure '{game_name}' is running.")
        return
    
    print("\n✅ Model loaded successfully!")
    print(f"\n🎯 Looking for game window: '{game_name}'")
    
    print("\n🔧 Focusing game window...")
    focus_window(window_id)
    print("✅ Window focused! (Keep game window in focus for best performance)")
    print("⏳ Starting agent in 2 seconds... (Press Ctrl+C to stop)")
    time.sleep(2)
    
    frame_count = 0
    decision_interval = 3
    goal_reached = False
    frames_not_facing = 0
    
    with mss.mss() as sct:
        try:
            while True:
                if goal_reached:
                    print("\n" + "="*80)
                    print("✅ GOAL REACHED! Agent paused.")
                    print("="*80)
                    new_goal = get_user_goal()
                    
                    if new_goal.lower() == 'quit':
                        print("\n👋 Exiting agent...")
                        break
                    
                    print(f"\n🎯 New goal set: {new_goal}")
                    agent.update_goal(new_goal)
                    goal_reached = False
                    frame_count = 0
                    frames_not_facing = 0
                    print("⏳ Resuming in 2 seconds...")
                    time.sleep(2)
                    continue
                
                region = get_window_geometry(game_name)
                
                if not region:
                    print("⚠ Game window not found! Retrying in 2 seconds...")
                    time.sleep(1)
                    continue
                
                screen = sct.monitors[1]
                region = validate_region(region, screen["width"], screen["height"])
                
                try:
                    sct_img = sct.grab(region)
                    img_np = np.array(sct_img)
                    
                    img_pil = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    
                    if frame_count % decision_interval == 0:
                        print(f"\n{'='*60}")
                        print(f"Frame: {frame_count}")
                        print("🤔 Analyzing scene...")
                        
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
                        
                        if action.goal_reached:
                            goal_reached = True
                            print("\n✨ ✅ GOAL REACHED! ✅ ✨")
                            continue
                        
                        if not action.facing_target:
                            frames_not_facing += 1
                            print(f"\n🔍 Target not visible (frame {frames_not_facing}) - AI searching...")
                            
                            # If AI didn't provide mouse movement, add smooth automatic search
                            if action.mouse_look.get('x', 0) == 0 and action.mouse_look.get('y', 0) == 0:
                                print(f"   🔄 Auto-search: rotating camera to find target")
                                action.mouse_look = {"x": 50, "y": 0}
                            
                            if action.mouse_look:
                                execute_mouse_look(action.mouse_look)
                        else:
                            frames_not_facing = 0
                            
                            if action.mouse_look:
                                execute_mouse_look(action.mouse_look)
                        
                        if action.action_sequence:
                            execute_action_sequence(action.action_sequence)
                    
                    display_img = cv2.resize(img_np, (800, 600))
                    cv2.putText(display_img, f"Frame: {frame_count}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    frame_count += 1
                    
                except Exception as e:
                    print(f"⚠ Capture/Processing failed: {e}")
                
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
