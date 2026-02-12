import time
from api.client import SimClient

def main():
    print("Initializing Client...")
    client = SimClient()
    
    print("Connecting to Simulation...")
    # This acts as the "Server" starter in a local env
    client.connect()
    
    print("Sending Drones to corners...")
    # Send all drones to (10, 10)
    for i in range(4):
        client.send_move_command(i, 10.0 + i*10, 50.0)
        
    try:
        start = time.time()
        frames = 0
        while time.time() - start < 5.0:
            state = client.get_latest_state()
            if state:
                frames += 1
                if frames % 60 == 0:
                    t = state['elapsed']
                    d0 = state['drones'][0]['pos']
                    print(f"Sim Time: {t:.2f}s | Drone 0 Pos: {d0}")
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        pass
    finally:
        print("Disconnecting...")
        client.disconnect()

if __name__ == "__main__":
    main()
