import gymnasium as gym
import numpy as np
from gymnasium import spaces
from core.simulation import GridSimulation
from core import config as cp

class IARCGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.sim = GridSimulation()
        self.render_mode = render_mode
        self.dt = 1.0 / 60.0
        
        # Observation Space: Needs to be defined based on what the agent sees
        # For now, let's just expose drone positions and velocities as a flat vector
        # 4 drones * (2 pos + 2 vel) = 16 floats
        # + Mines detected? + Grid?
        # Let's start simple: 16 floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cp.NUM_DRONES * 4,), dtype=np.float32
        )
        
        # Action Space: Velocity targets for 4 drones? Or Waypoints?
        # Let's say Move commands (x, y) for each drone
        # 4 drones * 2 coords = 8 floats
        # Bounds: Arena W/H
        self.action_space = spaces.Box(
            low=0.0, 
            high=max(cp.ARENA_WIDTH_FT, cp.ARENA_HEIGHT_FT), 
            shape=(cp.NUM_DRONES, 2), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = GridSimulation() # Reset logic
        return self._get_obs(), {}

    def step(self, action):
        # Action: (4, 2) array of target x, y
        for i, target in enumerate(action):
            self.sim.apply_command({
                "type": "MOVE",
                "id": i,
                "x": float(target[0]),
                "y": float(target[1])
            })
            
        # Step Physics
        # Maybe step multiple times if control freq < physics freq?
        # For now 1:1
        self.sim.step(self.dt)
        
        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = False
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        state = self.sim.get_state()
        # Flatten drone states
        obs_list = []
        for d in state["drones"]:
            obs_list.extend(d["pos"])
            obs_list.extend(d["vel"])
        return np.array(obs_list, dtype=np.float32)

    def _calculate_reward(self):
        # Placeholder reward
        return 0.0

    def render(self):
        if self.render_mode == "rgb_array":
            # TODO: Implement headless rendering if needed
            return np.zeros((100, 100, 3), dtype=np.uint8)
