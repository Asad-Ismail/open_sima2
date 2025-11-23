import time
import mss
import numpy as np
import cv2
from pynput.keyboard import Controller
from unsloth import FastLanguageModel
from PIL import Image

# 1. SETUP: Qwen2.5-VL (The Brain)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,
    gpu_memory_utilization=0.6, 
)
FastLanguageModel.for_inference(model)

# 2. SETUP: Controls
keyboard = Controller()
sct = mss.mss()
monitor = {"top": 0, "left": 0, "width": 1024, "height": 768} # Match your game resolution

# 3. THE VISUAL SENSOR (The Light Gem)
# In The Dark Mod, there is a gem at the bottom center of the screen.
# Bright = You are visible (Bad). Dark = You are hidden (Good).
def get_visibility_score(screenshot_np):
    h, w, _ = screenshot_np.shape
    # Crop the bottom center where the gem is
    gem_crop = screenshot_np[int(h*0.85):int(h*0.95), int(w*0.45):int(w*0.55)]
    # Calculate brightness (0 = Black/Hidden, 255 = White/Visible)
    brightness = np.mean(gem_crop)
    return brightness

print("Start The Dark Mod! AI taking over in 5s...")
time.sleep(5)

# 4. GAME LOOP
while True:
    # A. SEE
    sct_img = sct.grab(monitor)
    img_np = np.array(sct_img)
    img_pil = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    # B. SENSE (Visual Feedback)
    visibility = get_visibility_score(img_np)
    status = "HIDDEN (SAFE)" if visibility < 50 else "VISIBLE (DANGER)"
    
    # C. THINK (VLM)
    prompt = f"""
    You are playing a stealth game. 
    Status: {status} (Light Level: {int(visibility)}).
    Goal: Stay hidden and explore. If VISIBLE, find shadows immediately.
    What key should I press? (w, a, s, d, c=crouch).
    Output JSON: {{"reason": "Too bright, crouching and moving to corner", "key": "c"}}
    """
    
    # (Send to Qwen for inference...)
    # ...
    
    # D. REWARD SIGNAL (For Self-Improvement)
    # If you were recording data:
    # if visibility < 30: 
    #    save_frame(img_pil, "success_hidden") # REWARD +1
    # else:
    #    save_frame(img_pil, "fail_visible")   # REWARD -1
    
    print(f"AI Status: {status}")