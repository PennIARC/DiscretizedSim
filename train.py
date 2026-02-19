import time
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from gym_env import IARCGymEnv
from ppo_agent import PPOAgent
from core import config as cp

def main():
    # --- Configuration ---
    # Create unique run name based on time
    run_name = f"run_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir)
    print(f"TRAIN: Logging to {log_dir}")
    print(f"TRAIN: Run 'tensorboard --logdir=runs' to view live graphs.")

    # Initialize Headless Environment
    env = IARCGymEnv(render_mode=None) # Explicitly None to prevent pygame init
    
    # Initialize Agent
    # 35 Input Features -> 2 Continuous Outputs (dx, dy)
    agent = PPOAgent(obs_dim=env.obs_dim, action_dim=2, lr=3e-4)
    
    MAX_EPISODES = 5000
    # 7 Minutes at 60Hz = 25,200 steps
    MAX_STEPS_PER_EP = 25200 
    
    # How often to update the network (in total agent steps)
    # 4096 is a standard PPO batch size
    UPDATE_TIMESTEP = 4096 
    
    global_step = 0
    buffer_step_count = 0
    
    try:
        for episode in range(MAX_EPISODES):
            obs, _ = env.reset()
            episode_reward = 0
            
            # Tracking metrics for this episode
            ep_metrics = {
                "discovery": 0.0,
                "safety": 0.0,
                "lazy": 0.0,
                "collisions": 0
            }
            
            start_time = time.time()
            
            for step in range(MAX_STEPS_PER_EP):
                global_step += 1
                
                # 1. Select Action
                # obs is (4, 35). Agent returns (4, 2)
                actions, log_probs, values = agent.select_action(obs)
                
                # 2. Step Environment
                next_obs, rewards, term, trunc, info = env.step(actions)
                
                # 3. Store Data for Training
                # We treat each drone as an independent sample
                for i in range(cp.NUM_DRONES):
                    agent.store(
                        obs[i], 
                        actions[i], 
                        log_probs[i], 
                        rewards[i], 
                        (term or trunc), 
                        values[i]
                    )
                    buffer_step_count += 1
                
                # 4. Logging & Updates
                obs = next_obs
                episode_reward += np.sum(rewards)
                
                # Accumulate custom metrics from env info
                if "metrics" in info:
                    ep_metrics["discovery"] += info["metrics"]["discovery"]
                    ep_metrics["safety"] += info["metrics"]["safety"]
                    ep_metrics["lazy"] += info["metrics"]["lazy"]
                ep_metrics["collisions"] = info["collisions"] # This is cumulative in env
                
                # PPO Update
                if buffer_step_count >= UPDATE_TIMESTEP:
                    print(f"TRAIN: Updating Policy at step {global_step}...")
                    agent.update()
                    buffer_step_count = 0

                if term or trunc:
                    break
            
            # --- End of Episode Reporting ---
            duration = time.time() - start_time
            steps_per_sec = step / duration
            
            # Console Log
            print(f"Ep {episode} | Reward: {episode_reward:.1f} | Collisions: {ep_metrics['collisions']} | "
                  f"Scanned: {info['scanned_percent']:.1%} | SPS: {int(steps_per_sec)}")
            
            # TensorBoard Log (The Live Graphs)
            writer.add_scalar("Overview/Total_Reward", episode_reward, episode)
            writer.add_scalar("Overview/Map_Coverage", info['scanned_percent'], episode)
            writer.add_scalar("Safety/Total_Collisions", ep_metrics['collisions'], episode)
            
            # Detailed Reward Breakdown (Crucial for debugging)
            writer.add_scalar("Rewards/Discovery", ep_metrics['discovery'], episode)
            writer.add_scalar("Rewards/Safety_Penalty", ep_metrics['safety'], episode)
            writer.add_scalar("Rewards/Lazy_Penalty", ep_metrics['lazy'], episode)
            
            writer.add_scalar("System/Steps_Per_Second", steps_per_sec, episode)
            
            # Save Model every 10 episodes
            if episode % 10 == 0:
                torch.save(agent.policy.state_dict(), f"{log_dir}/model_{episode}.pth")

    except KeyboardInterrupt:
        print("TRAIN: Interrupted. Saving current model...")
        torch.save(agent.policy.state_dict(), f"{log_dir}/model_interrupted.pth")
        
    writer.close()
    env.close()

if __name__ == "__main__":
    main()