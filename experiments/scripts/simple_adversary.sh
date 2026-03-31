#!/bin/bash

# Training Script for Simple Adversary Scenario
# This script contains commands to run MADDPG experiments on the simple_adversary scenario
# using the MADDPG algorithm with optional diffusion models and ERNIE enhancements.

# Environment variables
export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

# =============================================================================
# SIMPLE_ADVERSARY SCENARIO
# =============================================================================
# Two-agent environment with one adversary and one good agent

echo "Starting Simple Adversary Experiments..."

# Standard training
echo "Training standard MADDPG on simple_adversary..."
python ../train.py --scenario simple_adversary --exp-name saRun1 --num-episodes 60000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for saRun1best..."
python ../train.py --scenario simple_adversary --exp-name saRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ../../saRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for saRun1..."
python ../train.py --scenario simple_adversary --exp-name saRun1 --mode train_diffusion --diffusion-data-path ../../saRun1_diffusion_data.npz --diffusion-model-path ../../saRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 1000

# Test with diffusion
echo "Testing saRun1best with diffusion model..."
python ../train.py --scenario simple_adversary --exp-name saRun1best --mode test --diffusion-model-path ../../saRun1_diffusion_model.pt

# # ERNIE-enhanced training
# echo "Training ERNIE-enhanced MADDPG on simple_adversary..."
# python ../train.py --scenario simple_adversary --exp-name pdeRun1 --num-episodes 60000 --save-rate 500 --use_ernie

# # Collect diffusion data for ERNIE
# echo "Collecting diffusion data for pdeRun1best..."
# python ../train.py --scenario simple_adversary --exp-name pdeRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./pdeRun1_diffusion_data.npz

# # Train diffusion for ERNIE
# echo "Training diffusion model for pdeRun1..."
# python ../train.py --scenario simple_adversary --exp-name pdeRun1 --mode train_diffusion --diffusion-data-path ./pdeRun1_diffusion_data.npz --diffusion-model-path ./pdeRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# # Test ERNIE with diffusion
# echo "Testing pdeRun1best with diffusion model..."
# python ../train.py --scenario simple_adversary --exp-name pdeRun1best --mode test --diffusion-model-path ./pdeRun1_diffusion_model.pt

# # Additional runs
# echo "Training pdSRRun2..."
# python ../train.py --scenario simple_adversary --exp-name pdSRRun2 --num-episodes 30000 --save-rate 500

# # Parallel testing
# echo "Running parallel tests for pdSRRun1..."
# python ../train.py --scenario simple_adversary --exp-name pdSRRun1 --mode test & python ../train.py --scenario simple_adversary --exp-name pdSRRun1best --mode test

echo "Simple Adversary experiments completed!"