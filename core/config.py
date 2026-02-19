
# --- Simulation Configuration ---
PX_PER_FOOT = 5.0            # Visual scale
ARENA_WIDTH_FT = 300.0
ARENA_HEIGHT_FT = 80.0
BACKGROUND_COLOR = (20, 25, 30)


# --- RL Specifics ---
RL_MAX_WAYPOINT_DIST = 10.0  # Max distance a single action can request
RL_HISTORY_LEN = 3           # Number of past frames to stack (handling delay)


# --- Physics & Rules ---
NUM_DRONES = 4
DRONE_RADIUS_FT = 0.5        
DETECTION_RADIUS_FT = 6.0   
MAX_SPEED_FT = 15.0          # REDUCED: Low top speed for reliable detection
MAX_ACCEL_FT = 80.0          # HIGH: Very snappy/responsive
TURN_RATE_RAD = 50.0    

# --- PID Control ---
PID_KP = 80.0
PID_KI = 10.0
PID_KD = 40.0
TICK_RATE = 1/60.0

# --- Map Generation ---
MINE_COUNT_MIN = 300
MINE_COUNT_MAX = 400
MINE_RADIUS_FT = 0.5
SAFE_DIST_FT = 1.0

# --- Visuals ---
VISUAL_DRONE_SIZE = 1.0      # Scale factor for drone drawing
GRID_LINE_SPACING = 50       # Feet

# --- Colors (Kept for compatibility, though core sim doesn't draw) ---
class Colors:
    detectionRadius = (177, 173, 221)
    drone = (255, 255, 255)
    wayPoint = (64, 128, 67)

    arenaBackground = (10, 12, 15)
    arenaBorder = (33, 38, 55)
    gridLines = (16, 18, 23)

    undetectedMines = (80, 90, 130)
    detectedMines = (255, 91, 93)

    # UI & Misc
    black = (0, 0, 0)
    white = (255, 255, 255)
    uiText = (200, 200, 200)

    # Heatmap (Unsaturated background theme)
    heatmapHot = (120, 50, 60)   # Muted Red
    heatmapCold = (20, 25, 35)   # Muted Blue/Grey (matches grid/bg)
