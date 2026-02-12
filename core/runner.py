import time
import threading
import queue
from core.simulation import GridSimulation

class AsyncSimRunner(threading.Thread):
    def __init__(self, command_queue, state_queue, tick_rate=60):
        super().__init__()
        self.command_queue = command_queue
        self.state_queue = state_queue
        self.tick_rate = tick_rate
        self.running = False
        self.sim = GridSimulation()
        self.lock = threading.Lock()
        
    def run(self):
        print(f"SIM: Starting Physics Engine at {self.tick_rate}Hz")
        self.running = True
        dt = 1.0 / self.tick_rate
        
        while self.running:
            start_time = time.time()
            
            # 1. Process Commands
            while not self.command_queue.empty():
                try:
                    cmd = self.command_queue.get_nowait()
                    self.sim.apply_command(cmd)
                except queue.Empty:
                    break
                    
            # 2. Step Physics
            with self.lock:
                self.sim.step(dt)
                
            # 3. Publish State 
            # Drop old state if exists to prevent lag build-up (Real-time behavior)
            snapshot = self.sim.get_state()
            
            if not self.state_queue.empty():
                try:
                    self.state_queue.get_nowait()
                except queue.Empty:
                    pass
            self.state_queue.put(snapshot)

            # 4. Sleep to maintain rate
            elapsed_work = time.time() - start_time
            sleep_time = dt - elapsed_work
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        print("SIM: Engine Stopped")

    def stop(self):
        self.running = False
