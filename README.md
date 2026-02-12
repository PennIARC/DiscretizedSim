# IARC Discretized Simulation

A scalable, decoupled grid-based simulation for IARC drone swarms. Supports headless execution for RL/ML and visualized execution for debugging.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start
### 1. Visualizer
Run the simulation with a real-time view:
```bash
python visualizer_client.py
```

### 2. Headless Mode
Run without graphics (useful for scripts/testing):
```bash
python run_example.py
```

### 3. Reinforcement Learning
Use the Gymnasium environment:

```python
import gymnasium as gym
from gym_env import IARCGymEnv

env = IARCGymEnv()
obs, info = env.reset()
# ... training loop ...
```

## Project Structure
For a detailed explanation of the architecture and files, see [project_overview.md](project_overview.md).
