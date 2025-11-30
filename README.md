# PPO Improved - Actor-Critic Implementation for CartPole Balance Task

This repository contains an implementation of Proximal Policy Optimization (PPO) for solving the CartPole balance task using the DeepMind Control Suite (DMC).

## Overview

This is an actor-critic reinforcement learning implementation that trains a policy to balance a cartpole.

## Repository Contents

- `ppo_improved.py` - Main PPO implementation with training and evaluation
- `test_ppo_improved.py` - Script to visualize trained policy
- `saved_models/` - Trained model checkpoints
  - `actor_improved_ep300.pth` - Final trained actor network
  - `critic_improved_ep300.pth` - Final trained critic network
- Learning curve plots:
  - `learning_curve_reward_improved.png` - Training and evaluation rewards (mean ± std)
  - `learning_curve_entropy_improved.png` - Policy entropy over time
  - `learning_curve_kl_improved.png` - KL divergence over time
- Training plots:
  - `training_reward_improved.png` - Training rewards per episode
  - `training_entropy_improved.png` - Training entropy per episode
  - `training_kl_improved.png` - Training KL divergence per episode
  - `training_value_loss_improved.png` - Value function loss per episode

## Installation

### Requirements

```bash
pip install torch numpy matplotlib dm-control opencv-python
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- DeepMind Control Suite (dm-control)
- OpenCV (for visualization)

## Usage

### Training

To train the PPO agent:

```bash
python ppo_improved.py
```

This will:
- Train with 3 seeds (0, 1, 2) for statistical significance
- Evaluate with seed 10
- Generate learning curves with mean ± standard deviation
- Save model checkpoints every 50 episodes
- Generate all plots automatically

**Training Configuration:**
- **Training Seeds**: [0, 1, 2]
- **Evaluation Seed**: 10
- **Max Episodes**: 310
- **Max Steps per Episode**: 1024
- **Evaluation Frequency**: Every 10 episodes
- **Evaluation Episodes**: 3

### Quick Test Mode

For faster testing, edit `ppo_improved.py` and set:
```python
QUICK_TEST = True
```

This reduces training to 1 seed, 20 episodes, and faster evaluation.

### Testing Trained Policy

To visualize the trained policy:

```bash
python test_ppo_improved.py
```

This will:
- Load the trained model from `saved_models/actor_improved_ep300.pth`
- Display a live visualization window
- Run 5 episodes showing the policy's behavior
- Press 'q' to quit, 'p' to pause

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Actor Learning Rate** | 2.8e-4 | Learning rate for policy network |
| **Critic Learning Rate** | 3.2e-4 | Learning rate for value network |
| **Clip Ratio** | 0.2 | PPO clipping parameter |
| **Gamma** | 0.99 | Discount factor |
| **Lambda** | 0.95 | GAE (Generalized Advantage Estimation) parameter |
| **Batch Size** | 80 | Batch size for policy updates |
| **Number of Epochs** | 10 | Number of training epochs per episode |
| **Target KL** | 0.018 | Maximum KL divergence before early stopping |
| **Entropy Coefficient** | 0.001 | Entropy regularization coefficient |
| **Max Gradient Norm** | 0.5 | Gradient clipping threshold |

## Results

The PPO Improved implementation achieves:
- **Training Rewards**: Mean rewards reaching 900-1000+ by episode 300
- **Evaluation Rewards**: Consistent high performance on evaluation seed
- **Stable Learning**: Smooth convergence with low variance across seeds
- **Policy Convergence**: Entropy decreases steadily, indicating policy convergence

### Learning Curves

The code generates comprehensive learning curves showing:
1. **Training Rewards**: Mean ± std across 3 training seeds
2. **Evaluation Rewards**: Mean ± std across evaluation runs
3. **Policy Entropy**: Measures exploration vs exploitation
4. **KL Divergence**: Tracks policy update magnitude
5. **Value Loss**: Critic network learning progress

## Algorithm Details

### PPO (Proximal Policy Optimization)

PPO is an on-policy actor-critic algorithm that:
- Uses clipped surrogate objective to prevent large policy updates
- Employs Generalized Advantage Estimation (GAE) for advantage computation
- Performs multiple epochs of training on collected data
- Includes early stopping based on KL divergence
- Uses entropy regularization to maintain exploration

### Key Features

- **On-policy Learning**: Uses data from current policy
- **Stable Updates**: Clipped objective prevents destructive updates
- **Sample Efficiency**: Multiple epochs per episode
- **Exploration**: Entropy regularization maintains exploration
- **Early Stopping**: KL divergence monitoring prevents over-updating

## Code Structure

```
ppo_improved.py
├── Actor Network (64-64 architecture)
├── Critic Network (64-64-1 architecture)
├── PPOImproved Class
│   ├── get_action() - Sample action from policy
│   ├── compute_gae() - Generalized Advantage Estimation
│   ├── update() - Policy and value function updates
│   ├── train() - Main training loop
│   ├── _evaluate_actor() - Evaluation function
│   └── save_model() - Save checkpoints
└── main() - Training with multiple seeds and plotting
```

All hyperparameters are within standard ranges used in RL literature.

## License

This project is for educational purposes as part of EEE 598 coursework.

## Author

Created for EEE 598 - Reinforcement Learning Assignment

## References

- Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
- DeepMind Control Suite documentation
- PyTorch documentation

