import math
import random
import numpy as np
from core import config as cp

# --- Constants ---
GRID_RES = 1.0  # 1 ft per cell
GRID_W = int(cp.ARENA_WIDTH_FT / GRID_RES)
GRID_H = int(cp.ARENA_HEIGHT_FT / GRID_RES)

# States
STATE_UNKNOWN = 0
STATE_KNOWN = 1
STATE_MINE = 2

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def update(self, error, dt):
        if dt <= 0: return 0.0
        
        self.integral += error * dt
        
        if self.first_run:
            derivative = 0.0
            self.first_run = False
        else:
            derivative = (error - self.prev_error) / dt
            
        self.prev_error = error
        
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

class Drone:
    def __init__(self, id, start_x, start_y):
        self.id = id
        self.pos = [float(start_x), float(start_y)]
        self.vel = [0.0, 0.0]
        self.acc = [0.0, 0.0]
        self.active = True
        self.heading = 0.0 
        self.manual_control = False
        
        self.pid_x = PIDController(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        self.pid_y = PIDController(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        self.waypoints = []

    def set_pid_params(self, kp, ki, kd):
        self.pid_x.kp = kp
        self.pid_x.ki = ki
        self.pid_x.kd = kd
        self.pid_y.kp = kp
        self.pid_y.ki = ki
        self.pid_y.kd = kd

    def update_immediate_destination(self, x, y):
        self.waypoints = [(float(x), float(y))]

    def update_physics(self, dt):
        if not self.active: return
        
        target = None
        if self.waypoints:
            target = self.waypoints[0]
            
        if target:
            error_x = target[0] - self.pos[0]
            error_y = target[1] - self.pos[1]
            
            acc_x = self.pid_x.update(error_x, dt)
            acc_y = self.pid_y.update(error_y, dt)
            
            current_acc_mag = math.sqrt(acc_x**2 + acc_y**2)
            if current_acc_mag > cp.MAX_ACCEL_FT:
                scale = cp.MAX_ACCEL_FT / current_acc_mag
                acc_x *= scale
                acc_y *= scale
                
            self.acc = [acc_x, acc_y]
            
            dist_to_target = math.sqrt(error_x**2 + error_y**2)
            if dist_to_target < 1.0:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.pid_x.reset()
                    self.pid_y.reset()
                    self.vel = [0.0, 0.0]
                    self.acc = [0.0, 0.0]
        else:
            self.acc = [0.0, 0.0]
            self.vel[0] *= 0.1 # Strong braking if target reached/lost
            self.vel[1] *= 0.1
            
        # Global Friction/Drag (Air Resistance)
        self.vel[0] *= 0.95
        self.vel[1] *= 0.95
            
        self.vel[0] += self.acc[0] * dt
        self.vel[1] += self.acc[1] * dt
        
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        if speed > cp.MAX_SPEED_FT:
            scale = cp.MAX_SPEED_FT / speed
            self.vel[0] *= scale
            self.vel[1] *= scale
            
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        
        self.pos[0] = max(0.0, min(cp.ARENA_WIDTH_FT, self.pos[0]))
        self.pos[1] = max(0.0, min(cp.ARENA_HEIGHT_FT, self.pos[1]))
        
        if speed > 0.1:
            self.heading = math.atan2(self.vel[1], self.vel[0])

class GridSimulation:
    def __init__(self):
        self.drones = []
        self.mines_truth = []
        self.mines_detected = []
        self.elapsed = 0.0
        
        # --- Knowledge Grid ---
        # 0: Unknown, 1: Known, 2: Mine
        self.grid = np.zeros((GRID_H, GRID_W), dtype=np.int8) 
        
        # Coordinate Meshgrid for Vectorization
        y_coords = np.arange(GRID_H)
        x_coords = np.arange(GRID_W)
        self.xx_grid, self.yy_grid = np.meshgrid(x_coords, y_coords)
        
        # Centerline Bias Map (Static)
        center_y = GRID_H / 2.0
        self.bias_map = 1.0 - (np.abs(self.yy_grid - center_y) / center_y)
        self.bias_map = np.clip(self.bias_map, 0.0, 1.0)

        # Control Params
        self.sense_horizon = 50.0 
        self.utility_threshold = 5.0
        self.drone_repulsion_dist = 300.0
        
        # --- Tuning Hyperparameters ---
        self.topo_base_weight = 10.0      
        self.gain_info_attraction = 10.0  
        self.gain_forward_drift = 1.0     
        self.gain_forward_bias = 1.0      
        self.gain_drone_repulsion = 200.0 
        self.gain_lane_bias = 50.0         
        self.waypoint_extend_dist = 4.0   
        
        # Initialize Drones
        for i in range(cp.NUM_DRONES):
            start_y = 20.0 + (i * 15.0) 
            self.drones.append(Drone(i, 5.0, start_y))
            
        self.generate_map()
        
    def generate_map(self):
        self.mines_truth = []
        self.mines_detected = []
        self.grid.fill(STATE_UNKNOWN)
        
        count = random.randint(cp.MINE_COUNT_MIN, cp.MINE_COUNT_MAX)
        for _ in range(count):
            mx = random.uniform(20, cp.ARENA_WIDTH_FT - 5)
            my = random.uniform(5, cp.ARENA_HEIGHT_FT - 5)
            self.mines_truth.append([mx, my])
            
    def step(self, dt):
        """Advances the simulation by dt seconds."""
        self.elapsed += dt
        
        # Plan Phase (Compute forces/waypoints)
        self.plan_paths()
        
        # Physics Phase
        for drone in self.drones:
            drone.update_physics(dt)
            
            # Scan for mines
            for mine in self.mines_truth:
                d_sq = (drone.pos[0] - mine[0])**2 + (drone.pos[1] - mine[1])**2
                if d_sq < cp.DETECTION_RADIUS_FT**2:
                     if mine not in self.mines_detected:
                         self.mines_detected.append(mine)
                         
    def apply_command(self, cmd):
        """
        Handles external commands.
        cmd: dict with 'type', 'id', etc.
        """
        if cmd['type'] == 'MOVE':
            d_id = cmd['id']
            if 0 <= d_id < len(self.drones):
                self.drones[d_id].manual_control = True
                self.drones[d_id].update_immediate_destination(cmd['x'], cmd['y'])

    def get_state(self):
        """Returns a deep copy of the state."""
        drone_states = []
        for d in self.drones:
            drone_states.append({
                "id": d.id,
                "pos": list(d.pos),
                "vel": list(d.vel),
                "acc": list(d.acc),
                "heading": d.heading,
                "waypoints": list(d.waypoints)
            })

        return {
            "elapsed": self.elapsed,
            "drones": drone_states,
            "mines_truth": [list(m) for m in self.mines_truth],
            "mines_detected": [list(m) for m in self.mines_detected],
            "grid": self.grid.copy() 
        }

    def update_grid(self):
        # 1. Mark Known Areas (Drones scanning)
        radius_cells = int(cp.DETECTION_RADIUS_FT)
        
        for drone in self.drones:
            dx = int(drone.pos[0])
            dy = int(drone.pos[1])
            
            x0 = max(0, dx - radius_cells)
            x1 = min(GRID_W, dx + radius_cells)
            y0 = max(0, dy - radius_cells)
            y1 = min(GRID_H, dy + radius_cells)
            
            view_slice = self.grid[y0:y1, x0:x1]
            # Don't overwrite mines
            view_slice[view_slice == STATE_UNKNOWN] = STATE_KNOWN

        # 2. Mark Mines
        for m in self.mines_detected:
            mx, my = int(m[0]), int(m[1])
            if 0 <= mx < GRID_W and 0 <= my < GRID_H:
                self.grid[my, mx] = STATE_MINE
                
    def get_frontier_utility(self):
        known_mask = (self.grid != STATE_UNKNOWN).astype(np.int8)
        unknown_mask = (self.grid == STATE_UNKNOWN).astype(np.int8)
        
        k_up = np.pad(known_mask[1:, :], ((0, 1), (0, 0)), mode='constant')
        k_down = np.pad(known_mask[:-1, :], ((1, 0), (0, 0)), mode='constant')
        k_left = np.pad(known_mask[:, 1:], ((0, 0), (0, 1)), mode='constant')
        k_right = np.pad(known_mask[:, :-1], ((0, 0), (1, 0)), mode='constant')
        
        neighbor_sum = k_up + k_down + k_left + k_right 
        
        utility_raw = (self.topo_base_weight ** neighbor_sum) * unknown_mask
        utility = utility_raw * self.bias_map
        
        utility = self.simple_blur(utility)
        utility = self.simple_blur(utility)
        
        return utility

    def simple_blur(self, grid):
        shifts = []
        weights = []
        
        # Center
        shifts.append(grid)
        weights.append(4.0)
        
        # Cardinal
        shifts.append(np.pad(grid[1:, :], ((0, 1), (0, 0)), mode='constant')) 
        shifts.append(np.pad(grid[:-1, :], ((1, 0), (0, 0)), mode='constant')) 
        shifts.append(np.pad(grid[:, 1:], ((0, 0), (0, 1)), mode='constant')) 
        shifts.append(np.pad(grid[:, :-1], ((0, 0), (1, 0)), mode='constant')) 
        weights.extend([2.0]*4)
        
        total = np.zeros_like(grid)
        div = 0.0
        
        for s, w in zip(shifts, weights):
            total += s * w
            div += w
            
        return total / div

    def plan_paths(self):
        self.update_grid()
        utility_map = self.get_frontier_utility()
        
        high_util_indices = np.argwhere(utility_map > self.utility_threshold)
        
        if len(high_util_indices) == 0:
            for drone in self.drones:
                if not drone.manual_control:
                    drone.update_immediate_destination(cp.ARENA_WIDTH_FT - 5, drone.pos[1])
            return

        high_util_coords = high_util_indices[:, [1, 0]].astype(np.float32) # [x, y]
        high_util_values = utility_map[high_util_indices[:, 0], high_util_indices[:, 1]] # utilities
        
        for drone in self.drones:
            if drone.manual_control:
                continue
                
            # A. Information Attraction
            d_pos = np.array(drone.pos, dtype=np.float32)
            
            diffs = high_util_coords - d_pos # (N, 2)
            dists_sq = np.sum(diffs**2, axis=1) # (N,)
            
            mask = dists_sq < (self.sense_horizon ** 2)
            
            gradX, gradY = 0.0, 0.0
            
            valid_diffs = diffs[mask]
            
            if len(valid_diffs) > 0:
                valid_dists = np.sqrt(dists_sq[mask])
                valid_utils = high_util_values[mask]
                
                weights = valid_utils / (valid_dists + 0.1)
                
                vectors = (valid_diffs / (valid_dists[:, None] + 0.1)) * weights[:, None]
                force = np.sum(vectors, axis=0) # (2,)
                
                gradX += force[0] * self.gain_info_attraction
                gradY += force[1] * self.gain_info_attraction
            else:
                  gradX += self.gain_forward_drift
            
            # B. Forward Bias 
            gradX += self.gain_forward_bias
            
            # C. Lane Bias 
            lane_h = cp.ARENA_HEIGHT_FT / cp.NUM_DRONES
            center_y = (drone.id * lane_h) + (lane_h / 2.0)
            gradY += (center_y - drone.pos[1]) * self.gain_lane_bias
            
            # D. Repulsion (Drones)
            for other in self.drones:
                if other != drone:
                    dx = drone.pos[0] - other.pos[0]
                    dy = drone.pos[1] - other.pos[1]
                    d2 = dx**2 + dy**2
                    if d2 < self.drone_repulsion_dist**2:
                        d = math.sqrt(d2) + 0.1
                        f = self.gain_drone_repulsion / d2 
                        gradX += (dx/d) * f
                        gradY += (dy/d) * f

            # E. Repulsion from Scanned Areas 
            gx, gy = int(drone.pos[0]), int(drone.pos[1])
            repel_rad = 6
            x0, x1 = max(0, gx - repel_rad), min(GRID_W, gx + repel_rad)
            y0, y1 = max(0, gy - repel_rad), min(GRID_H, gy + repel_rad)
            
            if x1 > x0 and y1 > y0:
                local = self.grid[y0:y1, x0:x1]
                ky, kx = np.where(local != STATE_UNKNOWN)
                if len(kx) > 0:
                    wx = (x0 + kx).astype(np.float32)
                    wy = (y0 + ky).astype(np.float32)
                    
                    dx = drone.pos[0] - wx
                    dy = drone.pos[1] - wy
                    d2 = dx**2 + dy**2 + 0.1
                    
                    repel_strength = 5.0
                    fx = np.sum((dx / d2)) * repel_strength
                    fy = np.sum((dy / d2)) * repel_strength
                    
                    gradX += fx
                    gradY += fy
                        
            # Normalize and Apply
            total_mag = math.sqrt(gradX**2 + gradY**2)
            if total_mag > 0.01:
                gradX /= total_mag
                gradY /= total_mag
                
            extend = self.waypoint_extend_dist
            tx = drone.pos[0] + gradX * extend
            ty = drone.pos[1] + gradY * extend
            
            ty = max(1.0, min(cp.ARENA_HEIGHT_FT - 1.0, ty))
            tx = max(0.0, min(cp.ARENA_WIDTH_FT - 1.0, tx))
            
            drone.update_immediate_destination(tx, ty)
