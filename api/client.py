import queue
from core.runner import AsyncSimRunner

class SimClient:
    def __init__(self):
        self.command_queue = queue.Queue()
        self.state_queue = queue.Queue()
        self.runner = None
        
    def connect(self):
        """Starts the background simulation thread."""
        if self.runner is not None and self.runner.is_alive():
            print("API: Already connected")
            return
            
        self.runner = AsyncSimRunner(self.command_queue, self.state_queue)
        self.runner.start()
        
    def disconnect(self):
        """Stops the simulation."""
        if self.runner:
            self.runner.stop()
            self.runner.join()
            
    def send_move_command(self, drone_id, x, y):
        """Sends a destination command to a specific drone."""
        cmd = {
            "type": "MOVE",
            "id": drone_id,
            "x": x,
            "y": y
        }
        self.command_queue.put(cmd)
        
    def get_latest_state(self):
        """
        Returns the most recent state snapshot from the sim.
        Returns None if no state is available yet.
        """
        try:
            return self.state_queue.get_nowait()
        except queue.Empty:
            return None
