# MADDPG Experiments - Reorganized Structure

This directory contains the reorganized MADDPG (Multi-Agent Deep Deterministic Policy Gradient) experiments codebase.

## 📁 Directory Structure

```
experiments/
├── core/                    # Core modules (organized from original train.py)
│   ├── __init__.py         # Module imports
│   ├── config.py           # Argument parsing and configuration
│   ├── environment.py      # Environment creation and utilities
│   ├── training.py         # Training functions
│   ├── testing.py          # Testing and evaluation functions
│   ├── noise.py            # Noise and perturbation utilities
│   └── diffusion.py        # Diffusion model utilities
├── scripts/                # Organized experiment scripts
│   ├── simple_adversary.sh
│   ├── simple_tag.sh
│   ├── simple_speaker_listener.sh
│   ├── simple_crypto.sh
│   ├── simple_push.sh
│   └── simple_spread.sh
├── train.py                # Original monolithic file (kept for reference)
├── train_new.py            # New modular entry point
└── README.md              # This file
```

## 🚀 Quick Start

### Activate Environment
```bash
conda activate maddpg
cd /home/axs0940/Rmaddpg/experiments/scripts
```

### Run Experiments
```bash
# Run simple adversary experiments
./simple_adversary.sh

# Run predator-prey experiments
./simple_tag.sh

# Or run individual components
cd ..
python train_new.py --mode train --scenario simple_adversary --exp-name saRun1
```

## 📋 Available Modes

- **`train`**: Train MADDPG agents
- **`test`**: Run robustness tests with noise sweeps
- **`collect_diffusion`**: Collect trajectories for diffusion training
- **`train_diffusion`**: Train diffusion denoising model

## 🔧 Key Improvements

### 1. **Modular Architecture**
- Split 1500+ line monolithic file into focused modules
- Each module has a single responsibility
- Easy to maintain and extend

### 2. **Organized Scripts**
- Separate bash scripts for each scenario
- Clear documentation and progress tracking
- Consistent structure across experiments

### 3. **Better File Management**
- Models saved to `../../models/` directory
- Results organized by scenario in `../../results/{scenario}/`
- Proper `.gitignore` configuration

### 4. **Cross-Machine Compatibility**
- Relative paths instead of absolute paths
- Works on any machine with proper directory structure

## 📊 Output Organization

```
Rmaddpg/
├── models/                 # Trained MADDPG models
├── results/                # CSV results by scenario
│   ├── simple_adversary/
│   ├── simple_tag/
│   └── ...
└── diffusion_data.npz      # Collected trajectories
```

## 🧩 Core Modules

### `config.py`
- Command-line argument parsing
- All experiment parameters and settings

### `environment.py`
- Environment creation utilities
- Agent trainer initialization
- Action space handling

### `training.py`
- MADDPG training loops
- Multi-run training with seeds
- CSV logging

### `testing.py`
- Robustness evaluation functions
- Noise perturbation testing
- Performance metrics

### `noise.py`
- Observation and action noise application
- Different noise distributions (Gaussian, uniform, shift)

### `diffusion.py`
- Diffusion model implementation
- Trajectory collection and training
- Action denoising inference

## 🔄 Migration Notes

- **Old**: `python train.py --mode train --scenario simple_adversary`
- **New**: `python train_new.py --mode train --scenario simple_adversary`

The original `train.py` is kept for reference but the new modular structure is recommended for all new experiments.