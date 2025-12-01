import time
import json
import os
import cv2
import numpy as np
import mss
from pynput import keyboard
from datetime import datetime
from PIL import Image
from screen_capture import get_window_geometry, validate_region

DATASET_DIR = "dataset/raw_playthrough"
os.makedirs(DATASET_DIR, exist_ok=True)

class ContinuousRecorder:
    def __init__(self, game_window="The Dark Mod"):
        self.current_keys = set()
        self.last_keys_captured = []  # Track keys from last capture
        self.recording = False
        self.frame_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.game_window = game_window
        
    def on_key_press(self, key):
        try:
            # Handle alphanumeric
            k = key.char.lower() if hasattr(key.char, 'lower') else key.char
        except AttributeError:
            # Handle special keys
            if key == keyboard.Key.space: k = 'space'
            elif key == keyboard.Key.enter: k = 'enter'
            elif key == keyboard.Key.up: k = 'up'
            elif key == keyboard.Key.down: k = 'down'
            elif key == keyboard.Key.left: k = 'left'
            elif key == keyboard.Key.right: k = 'right'
            elif key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r: k = 'c'
            else: k = None
            
        # Filter only keys we care about
        valid_keys = ['w', 's', 'a', 'd', 'c', 'space', 'enter', 'up', 'down', 'left', 'right']
        
        if k and k in valid_keys:
            self.current_keys.add(k)
            print(f"  Key pressed: {k} | Active: {self.current_keys}", end='\r')
            
        # Start/Stop with 'P'
        if hasattr(key, 'char') and key.char and key.char.lower() == 'p':
            self.toggle_recording()

    def on_key_release(self, key):
        try:
            k = key.char.lower() if hasattr(key.char, 'lower') else key.char
        except AttributeError:
            if key == keyboard.Key.space: k = 'space'
            elif key == keyboard.Key.enter: k = 'enter'
            elif key == keyboard.Key.up: k = 'up'
            elif key == keyboard.Key.down: k = 'down'
            elif key == keyboard.Key.left: k = 'left'
            elif key == keyboard.Key.right: k = 'right'
            elif key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r: k = 'c'
            else: k = None
            
        if k and k in self.current_keys:
            self.current_keys.discard(k)
            print(f"  Key released: {k} | Active: {self.current_keys}", end='\r')

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            print(f"\n🔴 RECORDING SESSION: {self.session_id}")
        else:
            print("\n⏸️ PAUSED")

    def start(self):
        print("⌨️  CONTROLS: Use Arrow Keys to Look, WASD to Move.")
        print("🔴 Press 'P' to Start/Pause Recording.")
        print(f"🎯 Capturing window: '{self.game_window}'")
        
        listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        listener.start()
        
        with mss.mss() as sct:
            try:
                while True:
                    if not self.recording:
                        time.sleep(0.1)
                        continue
                    
                    # 1. Get game window region
                    region = get_window_geometry(self.game_window)
                    if not region:
                        print(f"\n⚠️  Game window '{self.game_window}' not found! Waiting...")
                        time.sleep(2)
                        continue
                    
                    # Validate region bounds
                    screen = sct.monitors[1]
                    region = validate_region(region, screen["width"], screen["height"])
                    
                    # 2. Capture game window only
                    sct_img = sct.grab(region)
                    img = np.array(sct_img)
                    
                    # 3. Label (Placeholder Goal)
                    # Capture current keys at this exact moment
                    active_keys = sorted(list(self.current_keys))  # Sort for consistency
                    
                    # Only save frame if keys changed or every 3 frames
                    if active_keys != self.last_keys_captured or self.frame_count % 3 == 0:
                        if not active_keys: 
                            active_keys = ["none"]
                        
                        # We leave goal generic. The Post-Processor will fix this.
                        label_data = {
                            "action_sequence": active_keys[:5],
                            "mouse_look": {"x": 0, "y": 0}, # Unused now!
                            "goal_prompt": "UNLABELED" # Marker for post-processing
                        }
                        
                        # 4. Save
                        timestamp = int(time.time() * 1000)
                        base_name = f"{self.session_id}_{timestamp}"
                        
                        # Save RGB Image (Resize to 640x480 for speed/storage)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        img_resized = cv2.resize(img_rgb, (640, 480)) 
                        Image.fromarray(img_resized).save(os.path.join(DATASET_DIR, base_name + ".jpg"))
                        
                        with open(os.path.join(DATASET_DIR, base_name + ".json"), 'w') as f:
                            json.dump(label_data, f)
                        
                        self.last_keys_captured = active_keys
                        self.frame_count += 1
                        
                        # Clear line and show status
                        keys_str = ', '.join(active_keys) if active_keys != ["none"] else "none"
                        print(f"\r🔴 REC: {self.frame_count:04d} | Keys: [{keys_str}]" + " " * 30, end='')
                    
                    time.sleep(0.1)  # 10 FPS for better responsiveness
            except KeyboardInterrupt:
                print("\n✅ Saved.")

if __name__ == "__main__":
    rec = ContinuousRecorder()
    rec.start()