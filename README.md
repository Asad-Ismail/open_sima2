# Open SIMA 2

An open-source implementation of SIMA 2: An Agent that Plays, Reasons, and Learns in Virtual 3D Worlds.

[SIMA2 BLOG](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)

This project uses a vision-language model to control game characters autonomously, featuring:
- Real-time game frame analysis
- Structured action planning with reasoning
- Dynamic goal setting
- Mouse and keyboard control

![SIMA2 Training](imgs/siam2-train.png)

## Installation

### 1. System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential xdotool

# Fedora/RHEL
sudo dnf install python3-devel gcc xdotool

# Arch
sudo pacman -S python gcc xdotool
```

### 2. Python Environment

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# Install unsloth
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Usage

1. Start your game (tested with The Dark Mod)
2. Run the agent:
```bash
python agent.py
```

The agent will:
- Load the vision model
- Locate the game window
- Begin autonomous control after a countdown
- Allow you to set new goals dynamically

Press `Ctrl+C` to stop the agent.

## Project Structure

```
open_sima2/
├── agent.py              # Main agent entry point
├── src/
│   ├── vision_agent.py   # Vision-language model wrapper
│   ├── game_controls.py  # Keyboard/mouse control
│   └── screen_capture.py # Screen capture utilities
├── notebooks/            # Jupyter notebooks for experiments
├── imgs/                 # Images and documentation assets
├── pyproject.toml        # Project dependencies
└── README.md            # This file
```

## Features

- **Vision-Language Model**: Uses Qwen3-VL for understanding game state
- **Structured Output**: Pydantic models ensure reliable action parsing
- **Low Latency**: Optimized window management and input handling
- **Dynamic Goals**: Change objectives without restarting
- **Stealth Gameplay**: Visibility detection and threat awareness

## Requirements

- Python 3.9-3.13
- CUDA-capable GPU (for model inference)
- Linux with X11 (for window capture and control)
- xdotool (for input simulation)
