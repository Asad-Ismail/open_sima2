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
from src.game_controls import execute_action_sequence, focus_window


def get_user_goal() -> str:
    """Get new goal from user."""
    print("\n" + "="*80)
    print("💬 Enter new goal (or 'quit' to exit):")
    print("="*80)
    goal = input("🎯 Goal: ").strip()
    return goal


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous game agent")
    parser.add_argument("--model", type=str, 
                       default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
                       help="Model path (use 'models/gameplay_agent_lora' for fine-tuned)")
    parser.add_argument("--goal", type=str, default="Navigate to lantern",
                       help="Initial goal instruction")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎮 AUTONOMOUS GAME AGENT")
    print("=" * 80)
    print(f"\n🧠 Loading Vision Model: {args.model}")
    print(f"🎯 Initial Goal: {args.goal}")
    
    agent = VisionAgent(model_path=args.model)
    agent.update_goal(args.goal)
    
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
    goal_reached = False
    decision_interval = 3  # Make decision every 3 frames (0.3s)
    
    print("🎯 Using first action from each prediction (visually-grounded control)")
    
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
                    
                    # Make decision every decision_interval frames
                    if frame_count % decision_interval == 0:
                        print(f"\n{'='*60}")
                        print(f"Frame: {frame_count}")
                        print("🤔 Analyzing scene...")
                        
                        # Model predicts next 7 actions (for training temporal coherence)
                        predicted_actions = agent.predict_actions(img_pil)
                        
                        print(f"🎮 Predicted sequence: {predicted_actions}")
                        
                        # Check if goal reached (all actions are 'none')
                        if all(a == 'none' for a in predicted_actions):
                            goal_reached = True
                            print("\n✨ ✅ GOAL REACHED! ✅ ✨")
                            continue
                        
                        # Execute ONLY first action (based on current visual observation)
                        # The other 6 predictions are for training purposes only
                        # By next decision, we'll have new visual info and re-predict
                        current_action = predicted_actions[0]
                        
                        if current_action.lower() != 'none':
                            print(f"  ▶ Executing: {current_action}")
                            execute_action_sequence([current_action], hold_duration=0.3)
                        else:
                            # Wait this decision interval
                            time.sleep(0.3)
                    
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
