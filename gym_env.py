import gymnasium as gym
import numpy as np
import collections
import random
from gymnasium import spaces
from core.simulation import GridSimulation, STATE_UNKNOWN, STATE_KNOWN
from core import config as cp

class IARCGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.sim = GridSimulation()
        self.render_mode = render_mode
        self.dt = 1.0 / 60.0
        
        # 7 Minutes * 60 FPS = 25,200 steps
        self.max_steps = 25200 
        
        # --- Simulated Hardware Constants ---
        self.latency_buffer_size = 5  # Frames of delay (~80ms)
        self.packet_loss_rate = 0.05  # 5% command drop rate
        self.sensor_noise_std = 0.2   # 0.2 ft positional noise

        # --- Action Space (Relative Waypoints) ---
        # [dx, dy] in feet relative to current position
        # Normalized to [-1, 1], scaled by RL_MAX_WAYPOINT_DIST inside step
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cp.NUM_DRONES, 2), dtype=np.float32
        )

        # --- Observation Space ---
        # Self (4) + Neighbors (6) + Lidar (25) = 35 floats per drone
        self.obs_dim = 35
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cp.NUM_DRONES, self.obs_dim), dtype=np.float32
        )

        self.action_queues = [collections.deque(maxlen=self.latency_buffer_size) for _ in range(cp.NUM_DRONES)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = GridSimulation()
        self.steps = 0
        self.total_collisions = 0
        
        # Clear hardware buffers
        for q in self.action_queues:
            q.clear()
            for _ in range(self.latency_buffer_size):
                q.append(np.zeros(2))
                
        return self._get_obs(), {}

    def step(self, actions):
        self.steps += 1
        
        # 1. Simulate Hardware Layer (Latency + Loss)
        applied_actions = []
        for i, raw_action in enumerate(actions):
            # Packet Loss
            if random.random() > self.packet_loss_rate:
                self.action_queues[i].append(raw_action)
            
            # Latency: Pop oldest
            delayed_action = self.action_queues[i].popleft()
            
            # Maintain buffer size if empty
            if len(self.action_queues[i]) < self.latency_buffer_size:
                 self.action_queues[i].append(delayed_action)

            # Convert normalized [-1, 1] to physical relative offsets (Feet)
            # cp.RL_MAX_WAYPOINT_DIST should be in config, e.g., 10.0
            dx = delayed_action[0] * cp.RL_MAX_WAYPOINT_DIST
            dy = delayed_action[1] * cp.RL_MAX_WAYPOINT_DIST
            
            # Absolute Target
            curr_pos = self.sim.drones[i].pos
            target_x = curr_pos[0] + dx
            target_y = curr_pos[1] + dy
            
            applied_actions.append({
                "type": "MOVE", "id": i, "x": target_x, "y": target_y
            })

        # 2. Physics Step
        for cmd in applied_actions:
            self.sim.apply_command(cmd)
        self.sim.step(self.dt)

        # 3. Calculate Rewards & Obs
        obs = self._get_obs()
        rewards, metrics = self._calculate_rewards(actions)
        
        terminated = False
        truncated = self.steps >= self.max_steps
        
        # Info for TensorBoard
        total_cells = cp.ARENA_WIDTH_FT * cp.ARENA_HEIGHT_FT
        scanned_cells = np.count_nonzero(self.sim.grid != STATE_UNKNOWN)
        
        info = {
            "scanned_percent": scanned_cells / total_cells,
            "collisions": self.total_collisions,
            "metrics": metrics # Pass detailed reward breakdown
        }

        return obs, rewards, terminated, truncated, info

    def _get_obs(self):
        observations = []
        grid_h, grid_w = self.sim.grid.shape
        
        for i, drone in enumerate(self.sim.drones):
            # A. Self State (Noisy)
            nx = np.random.normal(0, self.sensor_noise_std)
            ny = np.random.normal(0, self.sensor_noise_std)
            
            # Normalize inputs roughly to [-1, 1] range for Neural Net stability
            self_vec = [
                (drone.pos[0] + nx) / cp.ARENA_WIDTH_FT,
                (drone.pos[1] + ny) / cp.ARENA_HEIGHT_FT,
                (drone.vel[0] / cp.MAX_SPEED_FT),
                (drone.vel[1] / cp.MAX_SPEED_FT)
            ]
            
            # B. Neighbor States (Relative & Noisy)
            neighbors = []
            for j, other in enumerate(self.sim.drones):
                if i == j: continue
                dx = (other.pos[0] - drone.pos[0]) + np.random.normal(0, self.sensor_noise_std)
                dy = (other.pos[1] - drone.pos[1]) + np.random.normal(0, self.sensor_noise_std)
                dist = np.sqrt(dx**2 + dy**2)
                neighbors.append((dist, dx, dy))
            
            # Sort by distance, take 3 closest
            neighbors.sort(key=lambda x: x[0])
            neighbor_vec = []
            for n in neighbors[:3]:
                # Normalize distance relative to arena diagonal approx
                neighbor_vec.extend([n[1]/100.0, n[2]/100.0]) 
            
            # Pad if < 3 neighbors
            while len(neighbor_vec) < 6:
                neighbor_vec.extend([0.0, 0.0])

            # C. Lidar / Grid Patch (5x5)
            # 0.0 = Unknown, 1.0 = Known/Mine
            patch = []
            cx, cy = int(drone.pos[0]), int(drone.pos[1])
            rad = 2 
            for r in range(-rad, rad+1):
                for c in range(-rad, rad+1):
                    px, py = cx + c, cy + r
                    if 0 <= px < grid_w and 0 <= py < grid_h:
                        val = 0.0 if self.sim.grid[py, px] == STATE_UNKNOWN else 1.0
                    else:
                        val = 1.0 # Walls are "Known"
                    patch.append(val)

            observations.append(np.array(self_vec + neighbor_vec + patch, dtype=np.float32))

        return np.array(observations)

    def _calculate_rewards(self, actions):
        """
        Calculates the Dense Reward Signal.
        Returns: (numpy array of rewards, dict of metrics for debugging)
        """
        rewards = np.zeros(cp.NUM_DRONES)
        
        # Metrics for TensorBoard analysis
        m_discovery = 0.0
        m_safety = 0.0
        m_lazy = 0.0
        
        # 1. Discovery Reward (Primary Objective)
        # We calculate exactly how many NEW pixels this drone revealed this frame.
        for i in range(cp.NUM_DRONES):
            new_cells = self.sim.update_sensor_coverage(i)
            # Reward: +0.1 per cell. A 10x10 area revealed = 10 points.
            # This is massive, incentivizing rapid movement into the unknown.
            r_disc = new_cells * 0.1
            rewards[i] += r_disc
            m_discovery += r_disc

        # 2. Safety Barrier (Collision Avoidance)
        # Quadratic Penalty
        warning_dist = 6.0 # ft
        critical_dist = 2.0 # ft
        
        for i, d1 in enumerate(self.sim.drones):
            min_dist = 999.0
            for j, d2 in enumerate(self.sim.drones):
                if i == j: continue
                dist = np.linalg.norm(np.array(d1.pos) - np.array(d2.pos))
                min_dist = min(min_dist, dist)
            
            # Apply Safety Gradient
            if min_dist < critical_dist:
                # CRITICAL: Crashed
                rewards[i] -= 100.0
                self.total_collisions += 1
                m_safety -= 100.0
            elif min_dist < warning_dist:
                # WARNING: Quadratic ramp
                # At 6.0ft: 0
                # At 2.1ft: -15.0 approx
                penalty = 1.0 * (warning_dist - min_dist)**2
                rewards[i] -= penalty
                m_safety -= penalty
                
            # 3. Lazy Penalty (Velocity)
            # If speed < 1.0 ft/s, small penalty. 
            # Prevents hovering in place to avoid safety penalties.
            speed = np.linalg.norm(d1.vel)
            if speed < 1.0:
                rewards[i] -= 0.05
                m_lazy -= 0.05
                
            # 4. Action Smoothing (Minimize jerk)
            # rewards[i] -= np.sum(np.square(actions[i])) * 0.01

        return rewards, {"discovery": m_discovery, "safety": m_safety, "lazy": m_lazy}