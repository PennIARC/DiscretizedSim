import time
import queue
from api.client import SimClient

def test_latency():
    client = SimClient()
    client.connect()
    
    print("Waiting for sim to warm up...")
    time.sleep(1.0)
    
    # Send a unique command (move to specific logic coord)
    # We'll monitor Drone 0 velocity change or position target
    
    # Let's use position target as trigger
    target_x = 123.45
    target_y = 67.89
    
    print(f"Sending Command: Move D0 to {target_x}, {target_y}")
    t0 = time.time()
    client.send_move_command(0, target_x, target_y)
    
    found = False
    details = []
    
    timeout = 2.0
    while time.time() - t0 < timeout:
        state = client.get_latest_state()
        if state:
            d0 = state['drones'][0]
            # Check waypoints
            if d0['waypoints']:
                wp = d0['waypoints'][0]
                if wp[0] == target_x and wp[1] == target_y:
                    t1 = time.time()
                    latency = (t1 - t0) * 1000.0
                    print(f"SUCCESS: Command reflected in state in {latency:.2f} ms")
                    found = True
                    break
        time.sleep(0.001) # Busy loopish
        
    if not found:
        print("FAILURE: Command not reflected within timeout.")
        
    client.disconnect()

if __name__ == "__main__":
    test_latency()
