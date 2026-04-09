#!/bin/bash

# Training Script for Simple Spread Scenario
# This script contains commands to run MADDPG experiments on the simple_spread scenario
# using the MADDPG algorithm with optional diffusion models and ERNIE enhancements.

# Environment variables
export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

# =============================================================================
# SIMPLE_SPREAD SCENARIO
# =============================================================================
# Cooperative spreading scenario

echo "Starting Simple Spread Experiments..."

# Standard training
echo "Training standard MADDPG on simple_spread..."
python ../train_new.py --scenario simple_spread --exp-name cnRun1 --num-episodes 60000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for cnRun1best..."
python ../train_new.py --scenario simple_spread --exp-name cnRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./cnRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for cnRun1..."
python ../train_new.py --scenario simple_spread --exp-name cnRun1 --mode train_diffusion --diffusion-data-path ./cnRun1_diffusion_data.npz --diffusion-model-path ./cnRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test with diffusion
echo "Testing cnRun1best with diffusion model..."
python ../train_new.py --scenario simple_spread --exp-name cnRun1best --mode test --diffusion-model-path ./cnRun1_diffusion_model.pt

# # Test another model
# echo "Testing cneRun1best with diffusion model..."
# python ../train_new.py --scenario simple_spread --exp-name cneRun1best --mode test --diffusion-model-path ./cneRun1_diffusion_model.pt

# # Additional training
# echo "Training cnSRRun1..."
# python ../train_new.py --scenario simple_spread --exp-name cnSRRun1 --num-episodes 60000 --save-rate 500

# # Parallel testing
# echo "Running parallel tests for cnSRRun1..."
# python ../train_new.py --scenario simple_spread --exp-name cnSRRun1 --mode test & python ../train_new.py --scenario simple_spread --exp-name cnSRRun1best --mode test

echo "Simple Spread experiments completed!"