# IARC Discretized Simulation - Project Overview

This project implements a discretized, grid-based simulation for the IARC drone competition. It is designed to be decoupled, allowing for headless execution (for Reinforcement Learning or data generation) and visualized execution (for debugging and demonstrations).

## Project Structure

```bash
IARCdiscretizedSim/
├── api/
│   ├── __init__.py
│   └── client.py          # User-facing API client (SimClient)
├── core/
│   ├── __init__.py
│   ├── config.py          # Configuration constants (Arena size, Physics, etc.)
│   ├── runner.py          # Thread-based simulation runner (AsyncSimRunner)
│   └── simulation.py      # Core physics engine (GridSimulation, Drone)
├── gym_env.py             # Gymnasium environment wrapper for RL
├── visualizer_client.py   # Pygame-based visualizer
├── run_example.py         # Headless execution example
├── latency_test.py        # Latency verification
├── verify_install.py      # Dependency check
└── requirements.txt       # Python dependencies
```

## Core Components

### 1. `core/simulation.py` (The Engine)
This file contains the pure Python simulation logic. It does **not** depend on Pygame or visualization libraries, making it fast and portable.

-   **`GridSimulation`**: The main class. It manages the drones, the grid state (known/unknown/mines), and advances physics.
    -   `step(dt)`: Advances the simulation by `dt` seconds.
    -   `apply_command(cmd)`: Handles external commands (e.g., `MOVE` drone).
    -   `get_state()`: Returns a dictionary snapshot of the current simulation state (drone positions, grid map, etc.).
-   **`Drone`**: Represents a single drone agent.
    -   Has internal PID controllers for movement.
    -   **Autonomous vs Manual**: By default, drones follow an internal exploration logic (`plan_paths`). If a `MOVE` command is sent via the API, the drone switches to `manual_control` mode and obeys the proper command until reset.

### 2. `core/runner.py` (The Runtime)
This handles the "Real-Time" aspect of the simulation.

-   **`AsyncSimRunner`**: Wraps `GridSimulation` in a background thread.
    -   It runs a loop at a fixed `TICK_RATE` (e.g., 60Hz).
    -   It manages thread-safe `command_queue` and `state_queue` for communication with the outside world.
    -   This allows your main script (or visualizer) to run at its own speed while the physics remain stable.

### 3. `api/client.py` (The Interface)
This is what you use to connect to a running simulation instance (if running in async mode).

-   **`SimClient`**:
    -   `connect()`: Starts the `AsyncSimRunner` thread.
    -   `send_move_command(id, x, y)`: Queues a command for the runner.
    -   `get_latest_state()`: Retrieves the most recent state snapshot from the updated queue.

---

## Execution Modes

### 1. Headless Simulation
**Use Case**: Running automated tests, data collection, or scripts where you don't need to see the output.

**How it works**:
1.  Your script instantiates `SimClient`.
2.  `client.connect()` spawns the `AsyncSimRunner` background thread.
3.  The runner steps the physics at 60Hz.
4.  Your script can sleep, do calculations, or send commands whenever it wants.
5.  State updates are retrieved via `client.get_latest_state()`.

**Example (`run_example.py`)**:
```python
client = SimClient()
client.connect()
# Send commands...
client.send_move_command(0, 50, 50)
# Loop and print state...
while True:
    state = client.get_latest_state()
    if state:
        print(state['drones'][0]['pos'])
```

### 2. Visualized Simulation
**Use Case**: Debugging behaviors, verifying path planning, demonstrations.

**How it works**:
1.  Run `python visualizer_client.py`.
2.  The script initializes Pygame for the window.
3.  It acts as a `SimClient` just like the headless script.
4.  It connects to the `AsyncSimRunner`.
5.  In the Pygame loop, it constantly fetches `get_latest_state()` and renders it to the screen.
6.  *Note:* The simulation logic still runs in the background thread. The visualization is just a "viewer" of the state stream.

---

## Gymnasium Environment

**Use Case**: Reinforcement Learning (RL) training using libraries like Stable Baselines 3, RLlib, etc.

**File**: `gym_env.py`

In this mode, we **do not** use the `AsyncSimRunner` or `SimClient`. RL training usually requires synchronous stepping (Agent acts -> Env steps -> Result returned), rather than real-time stepping.

### How it works
The `IARCGymEnv` class wraps `GridSimulation` directly.

-   **`reset()`**: Resets the internal `GridSimulation` instance.
-   **`step(action)`**:
    1.  Applies the `action` to the drones (e.g., velocity or position targets).
    2.  Calls `self.sim.step(dt)` manually.
    3.  Returns the new `observation`, `reward`, `terminated`, `truncated`, and `info`.

### Usage Example
```python
import gymnasium as gym
from gym_env import IARCGymEnv

# Initialize
env = IARCGymEnv()

# Observation Space: A flat array of drone states (positions, velocities)
# Action Space: (N_DRONES, 2) array for Target X, Y coordinates
obs, info = env.reset()

for _ in range(1000):
    # Sample random action (or use your agent model)
    action = env.action_space.sample()
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

### Key Differences for RL
-   **Synchronous**: The simulation time only advances when `env.step()` is called. This determines how fast the training runs (faster than real-time usually).
-   **Direct Access**: The environment accesses `self.sim` directly, avoiding queue overhead and threading complexity.
