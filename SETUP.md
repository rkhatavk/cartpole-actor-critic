# Quick Setup Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch numpy matplotlib dm-control opencv-python
```

## Running the Code

### Training
```bash
python ppo_improved.py
```

This will train for ~310 episodes with 3 seeds and generate all plots.

### Testing/Visualization
```bash
python test_ppo_improved.py
```

This will load the trained model and show a visualization window.

## Files Generated

After training, you'll have:
- Learning curve plots (with mean ± std)
- Training plots per episode
- Model checkpoints in `saved_models/`

## Notes

- Training takes time (several hours depending on hardware)
- Models are saved every 50 episodes
- Final model is at episode 300
- All plots are automatically generated

