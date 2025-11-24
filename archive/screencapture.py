import mss
import numpy as np
import cv2
import subprocess

def get_window_geometry(window_name_fragment):
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

with mss.mss() as sct:
    game_name = "The Dark Mod"
    
    while True:
        # 1. Get Dynamic Window Position
        region = get_window_geometry(game_name)
        
        if region:
            # Get your actual screen size (Monitor 1)
            screen = sct.monitors[1] 
            screen_w = screen["width"]
            screen_h = screen["height"]
            
            # Prevent 'Left' from being negative
            if region["left"] < 0: region["left"] = 0
            if region["top"] < 0: region["top"] = 0
            
            # Prevent 'Right' edge from exceeding Screen Width
            if (region["left"] + region["width"]) > screen_w:
                region["width"] = screen_w - region["left"]
                
            # Prevent 'Bottom' edge from exceeding Screen Height
            if (region["top"] + region["height"]) > screen_h:
                region["height"] = screen_h - region["top"]
            
            try:
                sct_img = sct.grab(region)
                img = np.array(sct_img)
                cv2.imshow("AI Eye", img)
            except Exception as e:
                print(f"Capture failed: {e}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()