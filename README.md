# 🚀 GridWorld Q-Learning Visualization

> **A beautiful, interactive, and fully open-source Q-Learning demo for GridWorld, with real-time terminal visualization and GIF export.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Features

- **Live Terminal Visualization**: See Q-table, V-table, and policy evolve in real time with [Rich](https://github.com/Textualize/rich)
- **GIF Export**: Save the entire learning process as a GIF for presentations or sharing
- **Configurable Grid Size**: Supports any N×N grid (default 5×5, easily switch to 10×10 or more)
- **Obstacle Support**: Add as many obstacles as you want, just by editing the config
- **Modular & Clean Code**: Easy to extend, hack, and learn from
- **Full Test Coverage**: Pytest-based tests for robust research and teaching
- **One-File Config**: All hyperparameters and environment settings in `config.yaml`

---

## 🖼️ Demo

![Q-Learning GIF Demo](fig/q_evolution.gif)

---

## 🏁 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/williamswang23/QVizGrid.git
cd QVizGrid
conda activate RL
pip install -r requirements.txt
```

### 2. Run Training with Visualization

```bash
python train.py
```

### 3. Run Tests

```bash
pytest -v
```

---

## ⚙️ Configuration

All settings are in `config.yaml`:

```yaml
env:
  grid_size: 10
  start: [0, 0]
  goal: [9, 9]
  holes:
    - [2, 3]
    - [5, 5]
    - [6, 2]
    - [7, 8]
  reward_goal: 1.0
  reward_hole: -1.0
  reward_step: -0.04

agent:
  alpha: 0.1
  gamma: 0.95
  epsilon_start: 0.9
  epsilon_end: 0.05
  epsilon_decay: 0.9995

train:
  max_episodes: 10000
  render_interval: 10
  snapshot_interval: 10
  convergence_tol: 1.0e-4
  convergence_patience: 20
  use_visualization: true
  create_gif: true

viz:
  fps: 5
```

---

## 📂 Project Structure

```
RL_Qvalue/
├── agent/           # Q-Learning agent
├── envs/            # GridWorld environment
├── viz/             # Visualization & GIF
├── tests/           # Pytest unit tests
├── outputs/         # GIF and frame outputs
├── docs/            # Experiment plans & docs
├── config.yaml      # All settings
├── train.py         # Main training script
├── requirements.txt # Dependencies
└── README.md        # This file
```

---

## 💡 Why This Project?

- **For Learners**: See how Q-Learning works, step by step, visually
- **For Teachers**: Use in class to explain RL concepts
- **For Researchers**: Prototype new ideas, test reward structures, or compare policies
- **For Everyone**: Open-source, hackable, and fun!

---

## 🧪 Testing

```bash
pytest -v
```

---

## 🤝 Contributing

Pull requests, issues, and suggestions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) if you want to help.

---

## 📄 License

Copyright (c) 2025, Williams.Wang. All rights reserved. Use restricted under LICENSE terms.

---

## 🙏 Acknowledgements

- [Rich](https://github.com/Textualize/rich) for terminal magic
- [NumPy](https://numpy.org/) for fast math
- [Matplotlib](https://matplotlib.org/) for GIFs
- [ImageIO](https://imageio.readthedocs.io/) for animation

---

**Star this repo if you like it! ⭐️** 