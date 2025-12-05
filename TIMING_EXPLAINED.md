# Action Execution and Timing

## How Recording Works

**Continuous Recorder (`src/continuos_recoder.py`):**
```
- Captures game screen at 10 FPS (every 0.1 seconds)
- Saves frames every 3 captures OR when keys change
- Records: which keys are pressed at that exact moment
- Result: ~0.3 second intervals between saved frames
```

**Example Recording:**
```
Time 0.0s: Frame 0 saved → Keys: ["w", "left"]
Time 0.1s: Frame 1 (not saved, same keys)
Time 0.2s: Frame 2 (not saved, same keys)
Time 0.3s: Frame 3 saved → Keys: ["w"]
Time 0.4s: Frame 4 (not saved, same keys)
Time 0.5s: Frame 5 (not saved, same keys)
Time 0.6s: Frame 6 saved → Keys: ["w", "right"]
```

## How Training Works

**Label Segments (`src/label_segments.py`):**
```
- Groups frames into 4-second segments (~40 frames)
- For each frame, creates sliding window of next 7 actions
- Example for frame N:
  actions = [action_N, action_N+1, action_N+2, ..., action_N+6]
```

**Sliding Window:**
```
Frame 0: ["w", "left", "w", "w", "right", "none", "none"]
Frame 1: ["left", "w", "w", "right", "none", "none", "up"]
Frame 2: ["w", "w", "right", "none", "none", "up", "w"]
```

The **first action** is always the action for the current frame!

## How Agent Executes

**Agent (`agent.py`):**
```python
decision_interval = 3  # Make decision every 3 frames (0.3s)

1. Capture frame
2. Every 3 frames:
   - Run model prediction → get 7 actions
   - Take ONLY the first action
   - Execute it for 0.3 seconds
3. Repeat
```

**Execution Function (`game_controls.py`):**
```python
def execute_action_sequence(actions, hold_duration=0.3):
    # Press all keys simultaneously (like human)
    for key in actions:
        xdotool keydown <key>
    
    # Hold for duration
    sleep(hold_duration)  # 0.3s = 3 frames
    
    # Release all keys
    for key in actions:
        xdotool keyup <key>
```

## Key Design Choices

### ✅ Simultaneous Key Execution
**Why:** Humans press multiple keys at once (e.g., "w" + "left" = move forward while turning)

**Before (WRONG):**
```python
Press 'w' → hold 0.3s → release → wait 0.2s
Press 'left' → hold 0.3s → release → wait 0.2s
Total: 1.0 second for 2 keys (TOO SLOW!)
```

**After (CORRECT):**
```python
Press 'w' AND 'left' simultaneously → hold 0.3s → release both
Total: 0.3 seconds for 2 keys (MATCHES RECORDING!)
```

### ✅ Hold Duration = 0.3s
**Why:** Matches the decision interval (3 frames @ 10 FPS)

This ensures:
- Actions execute for same duration as recorded
- Smooth continuous control
- No gaps or overlaps in execution

### ✅ Execute Only First Action
**Why:** Model predicts 7 future actions, but we only act on current frame

**Training:** Each frame learns to predict its own action + 6 future actions
**Inference:** We only execute the first prediction, then re-evaluate

This provides:
- Reactive behavior (can adapt to unexpected situations)
- No commitment to long action sequences
- Model constantly re-evaluating based on latest visual input

## Timing Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ RECORDING (Human Gameplay)                                  │
├─────────────────────────────────────────────────────────────┤
│ Frame 0 (0.0s): Keys=[w,left]  ← SAVED                      │
│ Frame 1 (0.1s): Keys=[w,left]                               │
│ Frame 2 (0.2s): Keys=[w,left]                               │
│ Frame 3 (0.3s): Keys=[w]       ← SAVED                      │
│ Frame 4 (0.4s): Keys=[w]                                    │
│ Frame 5 (0.5s): Keys=[w]                                    │
│ Frame 6 (0.6s): Keys=[w,right] ← SAVED                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ INFERENCE (Agent Playing)                                   │
├─────────────────────────────────────────────────────────────┤
│ Frame 0 (0.0s): Predict→[w,left,...] Execute: w+left (0.3s)│
│ Frame 1 (0.1s): (no decision)                               │
│ Frame 2 (0.2s): (no decision)                               │
│ Frame 3 (0.3s): Predict→[w,...] Execute: w (0.3s)          │
│ Frame 4 (0.4s): (no decision)                               │
│ Frame 5 (0.5s): (no decision)                               │
│ Frame 6 (0.6s): Predict→[w,right,...] Execute: w+right (0.3s)│
└─────────────────────────────────────────────────────────────┘
```

## Summary

**No special delays needed!** The system is designed to match:
- Recording captures state every 0.3s (saved frames)
- Agent decides every 0.3s (decision_interval=3)
- Agent executes for 0.3s (hold_duration=0.3)
- Keys pressed simultaneously (like human input)

The "none" actions in predictions naturally handle pauses/waiting - model learns when to output "none" for frames where no action should be taken.
