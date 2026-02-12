from gymnasium.utils.env_checker import check_env
from gym_env import IARCGymEnv

def run_check():
    print("Initializing Gym Env...")
    env = IARCGymEnv()
    
    print("Running check_env...")
    # This checks for API compliance
    check_env(env)
    print("Gym Env Check Passed!")
    
    print("Running Reset/Step Loop...")
    obs, info = env.reset()
    print(f"Initial Obs Shape: {obs.shape}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
    print("Loop finished successfully.")

if __name__ == "__main__":
    run_check()
