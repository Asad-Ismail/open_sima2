

We will implement open SIMA 2: An Agent that Plays, Reasons, and Learns With You in Virtual 3D Worlds.

[SIMA2 BLOG](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)

We will run a much smaller vision language reasoning model for this so we will be able to replicate their self improvement pipeline without costing lot of dollars.

![SIMA2 Training](imgs/siam2-train.png)

## Installation

### 1. Install System Dependencies

```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# On Fedora/RHEL
sudo dnf install python3-devel gcc

# On Arch
sudo pacman -S python gcc
```

### 2. Install uv (Fast Python Package Manager)

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### 3. Install Python Dependencies

```bash
# Create a virtual environment
uv venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate  # On Windows

# Install dependencies from pyproject.toml
uv sync

# Install unsloth (includes compatible xformers automatically)
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Usage

1. Start your game (e.g., The Dark Mod)
2. Run the agent:
```bash
python agentplay.py
```

The AI will take control after a 5-second countdown.


