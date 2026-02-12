import pygame
import sys
import numpy as np
import time
from api.client import SimClient
from core import config as cp

# Constants for Viz
SCREEN_W = 1200
SCREEN_H = int(SCREEN_W * (cp.ARENA_HEIGHT_FT / cp.ARENA_WIDTH_FT))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("IARC Sim V1 - Client Visualizer (API)")
    clock = pygame.time.Clock()
    
    # fonts
    font = pygame.font.SysFont("Arial", 18)
    
    # Start Sim
    print("Starting API Client...")
    sim = SimClient()
    sim.connect()
    
    running = True
    
    try:
        while running:
            # Event Loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # Example: Send command on click or key
                    if event.key == pygame.K_SPACE:
                        print("Sending manual command...")
                        sim.send_move_command(0, 50, 50)

            # Get State
            state = sim.get_latest_state()
            
            # Render
            screen.fill(cp.Colors.arenaBackground)
            
            if state:
                # 1. Draw Grid (Heatmap)
                if 'grid' in state:
                    grid = state['grid'] # numpy array
                    
                    # Manual colormap (Cold -> Hot)
                    # 0: Unknown (Gray), 1: Known (Blueish), 2: Mine (Red)
                    
                    # Create RGB Image
                    # (H, W, 3)
                    h, w = grid.shape
                    # Transpose for Pygame (W, H)
                    grid_t = grid.T 
                    
                    # Map colors
                    # Create empty RGB array
                    rgb = np.zeros((w, h, 3), dtype=np.uint8)
                    
                    # Fast boolean indexing
                    rgb[grid_t == 0] = (30, 30, 30)
                    rgb[grid_t == 1] = (50, 50, 100)
                    rgb[grid_t == 2] = (200, 50, 50)
                    
                    surf = pygame.surfarray.make_surface(rgb)
                    surf = pygame.transform.scale(surf, (SCREEN_W, SCREEN_H))
                    screen.blit(surf, (0, 0))

                # 2. Draw Mines (Truth)
                scale_x = SCREEN_W / cp.ARENA_WIDTH_FT
                scale_y = SCREEN_H / cp.ARENA_HEIGHT_FT
                
                for m in state['mines_truth']:
                    mx, my = m
                    pygame.draw.circle(screen, cp.Colors.undetectedMines, (int(mx * scale_x), int(my * scale_y)), 3)
                
                # detected mines
                for m in state['mines_detected']:
                    mx, my = m
                    pygame.draw.circle(screen, cp.Colors.detectedMines, (int(mx * scale_x), int(my * scale_y)), 3)

                # 3. Draw Drones
                for d in state['drones']:
                    dx, dy = d['pos']
                    sx = int(dx * scale_x)
                    sy = int(dy * scale_y)
                    
                    # Body
                    pygame.draw.circle(screen, cp.Colors.drone, (sx, sy), 5)
                    # ID
                    txt = font.render(str(d['id']), True, cp.Colors.white)
                    screen.blit(txt, (sx-4, sy-4))
                    
                    # Waypoint line
                    if d['waypoints']:
                        wx, wy = d['waypoints'][0]
                        pygame.draw.line(screen, cp.Colors.wayPoint, (sx, sy), (int(wx*scale_x), int(wy*scale_y)), 1)
                        
                # UI
                infos = [
                    f"Sim Time: {state['elapsed']:.2f}",
                    f"TPS: 60Hz Target",
                    f"Drones: {len(state['drones'])}"
                ]
                
                for i, t in enumerate(infos):
                    ts = font.render(t, True, cp.Colors.uiText)
                    screen.blit(ts, (10, 10 + i*20))
            else:
                txt = font.render("Waiting for Sim...", True, cp.Colors.uiText)
                screen.blit(txt, (SCREEN_W//2 - 50, SCREEN_H//2))

            pygame.display.flip()
            clock.tick(60)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing...")
        sim.disconnect()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
