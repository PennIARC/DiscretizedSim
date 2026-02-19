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
        
        # Internal PID for "Fake" flight controller physics
        self.pid_x = PIDController(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        self.pid_y = PIDController(cp.PID_KP, cp.PID_KI, cp.PID_KD)
        self.waypoints = []

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
            # Hover braking
            self.acc = [0.0, 0.0]
            self.vel[0] *= 0.1 
            self.vel[1] *= 0.1
            
        # Drag
        self.vel[0] *= 0.95
        self.vel[1] *= 0.95
            
        self.vel[0] += self.acc[0] * dt
        self.vel[1] += self.acc[1] * dt
        
        # Hard Speed Cap (Physics constraint)
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        if speed > cp.MAX_SPEED_FT:
            scale = cp.MAX_SPEED_FT / speed
            self.vel[0] *= scale
            self.vel[1] *= scale
            
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        
        # Bounds Check
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
        
        # 0: Unknown, 1: Known, 2: Mine
        self.grid = np.zeros((GRID_H, GRID_W), dtype=np.int8) 
        
        # Initialize Drones spaced out slightly
        for i in range(cp.NUM_DRONES):
            start_y = 10.0 + (i * 15.0) 
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
        
        # Physics Phase
        for drone in self.drones:
            drone.update_physics(dt)
            
            # Simple Mine Detection Logic (Perfect sensor within small radius)
            # Note: The Grid "Fog of War" update happens in get_sensor_update() now
            # so the RL agent can get the reward for it.
            
            # Check mines (Ground Truth check)
            for mine in self.mines_truth:
                d_sq = (drone.pos[0] - mine[0])**2 + (drone.pos[1] - mine[1])**2
                if d_sq < cp.DETECTION_RADIUS_FT**2:
                     if mine not in self.mines_detected:
                         self.mines_detected.append(mine)
                         
    def apply_command(self, cmd):
        if cmd['type'] == 'MOVE':
            d_id = cmd['id']
            if 0 <= d_id < len(self.drones):
                self.drones[d_id].manual_control = True
                self.drones[d_id].update_immediate_destination(cmd['x'], cmd['y'])

    def get_state(self):
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

    def update_sensor_coverage(self, drone_index):
        """
        Calculates which cells this specific drone is revealing RIGHT NOW.
        Returns: The number of *previously unknown* cells that just became known.
        """
        drone = self.drones[drone_index]
        radius_cells = int(cp.DETECTION_RADIUS_FT / GRID_RES)
        
        cx, cy = int(drone.pos[0]), int(drone.pos[1])
        
        x0 = max(0, cx - radius_cells)
        x1 = min(GRID_W, cx + radius_cells)
        y0 = max(0, cy - radius_cells)
        y1 = min(GRID_H, cy + radius_cells)
        
        # Slice of the global grid
        view_slice = self.grid[y0:y1, x0:x1]
        
        # Create a circular mask for the sensor
        y, x = np.ogrid[y0:y1, x0:x1]
        mask = (x - cx)**2 + (y - cy)**2 <= radius_cells**2
        
        # Find cells that are in the circle AND are currently Unknown (0)
        # We only care about revealing UNKNOWN cells for reward
        newly_revealed_mask = (view_slice == STATE_UNKNOWN) & mask
        new_cells_count = np.count_nonzero(newly_revealed_mask)
        
        # Update the global grid to Known (1)
        # Note: We don't overwrite Mines (2) because logic elsewhere handles that
        view_slice[newly_revealed_mask] = STATE_KNOWN
        
        return new_cells_count