"""
Test script for visualizing trained PPO Improved policy.
This script loads a saved model and displays the policy's behavior in the CartPole environment.
"""
import torch
import numpy as np
import cv2
from dm_control import suite
from ppo_improved import PPOImproved


def extract_state_from_observation(time_step):
    """Extract state vector from DeepMind Control Suite time step observation."""
    observation_dict = time_step.observation
    state_components = []
    # Sort keys for consistent ordering
    for key in sorted(observation_dict.keys()):
        value = observation_dict[key]
        state_components.append(value.flatten())
    return np.concatenate(state_components)


def run_policy_visualization(actor_network, domain='cartpole', task='balance', episodes=5):
    """
    Visualize the trained policy by running episodes in the environment.
    
    Args:
        actor_network: Trained actor network (PyTorch model)
        domain: Domain name for DeepMind Control Suite
        task: Task name for DeepMind Control Suite
        episodes: Number of episodes to run
    """
    environment = suite.load(domain_name=domain, task_name=task)
    window_width, window_height = 640, 480
    window_title = 'CartPole Balance - PPO Improved'
    
    for episode_num in range(episodes):
        current_time_step = environment.reset()
        current_state = extract_state_from_observation(current_time_step)
        total_reward = 0.0
        step_count = 0
        
        while not current_time_step.last():
            # Get action from policy (deterministic - use mean only)
            with torch.no_grad():
                action_mean, _ = actor_network(current_state)
                action = action_mean.squeeze().cpu().numpy()
            
            # Step environment
            current_time_step = environment.step(action)
            current_state = extract_state_from_observation(current_time_step)
            
            # Accumulate reward
            step_reward = current_time_step.reward if current_time_step.reward is not None else 0.0
            total_reward += step_reward
            step_count += 1
            
            # Render and display
            rendered_frame = environment.physics.render(
                height=window_height, 
                width=window_width, 
                camera_id=0
            )
            cv2.imshow(window_title, cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
            
            # Handle user input
            user_key = cv2.waitKey(1)
            if user_key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
            elif user_key & 0xFF == ord('p'):
                cv2.waitKey(0)  # Pause until key press
        
        print(f"Episode {episode_num + 1}: Reward={total_reward:.2f}, Steps={step_count}")
    
    cv2.destroyAllWindows()
    print(f"\nVisualization complete! Ran {episodes} episode(s).")


def main():
    """Main function to load model and run visualization."""
    # Configuration
    model_checkpoint = 'saved_models/actor_improved_ep300.pth'
    domain = 'cartpole'
    task = 'balance'
    num_episodes = 5
    
    print("=" * 60)
    print("PPO Improved Policy Visualization")
    print("=" * 60)
    print(f"Loading checkpoint: {model_checkpoint}\n")
    
    # Initialize agent and load trained model
    agent = PPOImproved(domain_name=domain, task_name=task)
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    agent.actor.load_state_dict(checkpoint['model_state_dict'])
    agent.actor.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    print("Starting visualization...")
    print("Controls: Press 'q' to quit, 'p' to pause\n")
    
    # Run visualization
    run_policy_visualization(agent.actor, domain, task, num_episodes)


if __name__ == "__main__":
    main()

