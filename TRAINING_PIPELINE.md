# Training Pipeline for Gameplay Agent

## Overview
This document explains the complete pipeline from data collection to playing with a fine-tuned model.

## Pipeline Steps

### 1. Data Collection
Record human gameplay using arrow keys for camera control:

```bash
cd src
python continuos_recoder.py
```

**Controls during recording:**
- `P` - Start/stop recording
- Arrow keys (`up`, `down`, `left`, `right`) - Camera control
- `w`, `a`, `s`, `d` - Movement
- `c` - Crouch
- `space` - Jump
- `enter` - Interact

**Output:** Raw frames + action labels in `dataset/raw_playthrough/`

### 2. Labeling
Generate goal instructions and create training data:

```bash
python src/label_segments.py
```

**What it does:**
- Segments recordings into 4-second clips (~40 frames at 10 FPS)
- Uses Qwen VLM to extract short goal instructions (e.g., "Go to lantern", "Check the posters")
- Creates sliding window labels: predicts next 7 actions from each frame
- Saves in Unsloth training format

**Output:** Labeled training data in `dataset/labeled_segments/`

**Training format:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "frame_0000.jpg"},
        {"type": "text", "text": "Goal: Go to the lantern\n\nPredict the next 7 actions:"}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"actions\": [\"w\", \"right\", \"w\", \"w\", \"none\", \"none\", \"none\"]}"
    }
  ]
}
```

### 3. Training
Fine-tune Qwen3-VL-8B with LoRA:

```bash
python train_vision_agent.py
```

**Training configuration:**
- Base model: `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`
- LoRA: r=16, alpha=16, dropout=0.05
- 4-bit quantization with gradient checkpointing
- Batch size: 2, Gradient accumulation: 4 (effective batch size: 8)
- Learning rate: 2e-4
- Epochs: 3

**Output:** Fine-tuned model saved to `models/gameplay_agent_lora/`

### 4. Inference
Play the game with your fine-tuned model:

**Using fine-tuned model:**
```bash
python agent.py --model models/gameplay_agent_lora --goal "Go to the lantern"
```

**Using base model (for comparison):**
```bash
python agent.py --model unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit --goal "Go to the lantern"
```

## Key Design Decisions

### Arrow Keys Instead of Mouse
- **Recording:** Captures arrow key presses (up/down/left/right)
- **Training:** Model learns to predict arrow keys in action sequence
- **Inference:** Arrow keys are executed via `execute_action_sequence()`
- **No mouse_look needed:** Camera control is discrete (arrow keys) not continuous (mouse deltas)

### Simple Action-Only Format
- **Training:** Only predicts `{"actions": [...]}`
- **No reasoning, no complex structure:** Faster inference, easier to train
- **7 actions per frame:** Sliding window of future actions

### Valid Actions
- Movement: `w`, `s`, `a`, `d`
- Camera: `up`, `down`, `left`, `right`
- Other: `c` (crouch), `space` (jump), `enter` (interact)
- Placeholder: `none` (no action)

## Data Requirements

### Minimum Training Data
- **1 hour recording** = ~900 segments (4s each) = ~36,000 training samples
- **Recommended:** 2-3 hours of diverse gameplay
- **Scenarios to cover:**
  - Navigation (indoor/outdoor)
  - Climbing/jumping
  - Door opening
  - Searching for objects
  - Avoiding guards

### Segment Duration Trade-off
- **4 seconds chosen:** Long enough for complete micro-goals (search → orient → move)
- Produces cleaner single-goal labels
- Better quality than 2.5s segments

## Model Performance

### Expected Behavior After Training
- Model predicts next 7 actions given current frame + goal
- Actions include both movement and camera control (via arrow keys)
- All actions are 'none' when goal is reached

### Testing
1. Record 30-60 minutes of gameplay
2. Label the data
3. Train for 3 epochs
4. Test with different goals:
   - "Go to the lantern"
   - "Climb the stairs"
   - "Open the door"
   - "Check the posters"

## Troubleshooting

### Model Not Moving Camera
- Check that arrow keys are in recorded action sequences
- Verify training data has diverse camera movements
- Try temperature=0.7 for more exploration

### Model Stuck
- Increase diversity of training scenarios
- Collect data with obstacles and complex navigation
- Add more varied goals during labeling

### Poor Goal Following
- Ensure goal instructions match training format (short, 3-5 words)
- Check that goals are realistic (present in training data)
- Try recording more examples of that specific goal

## Files Modified

### Core Pipeline
- `src/continuos_recoder.py` - Data collection
- `src/label_segments.py` - Labeling with VLM
- `train_vision_agent.py` - LoRA fine-tuning
- `agent.py` - Inference with trained model

### VisionAgent Changes
- Removed complex structured output (`GameAction` with reasoning, threats, etc.)
- Simple `predict_actions()` returns list of 7 actions
- Matches training format exactly
- Supports loading fine-tuned models

### Agent.py Changes
- Removed `mouse_look` execution (using arrow keys instead)
- Simplified to just action sequence prediction and execution
- Added `--model` and `--goal` command-line arguments
- Goal reached when all actions are 'none'

## Next Steps

1. **Collect diverse data:** Record 2-3 hours covering different scenarios
2. **Label segments:** Run labeling script to generate training data
3. **Train model:** Fine-tune Qwen3-VL with your data
4. **Test & iterate:** Try different goals, collect more data where needed
5. **Optimize:** Adjust LoRA hyperparameters, segment duration, etc.
