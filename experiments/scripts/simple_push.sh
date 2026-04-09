#!/bin/bash

# Training Script for Simple Push Scenario
# This script contains commands to run MADDPG experiments on the simple_push scenario
# using the MADDPG algorithm with optional diffusion models and ERNIE enhancements.

# Environment variables
export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

# =============================================================================
# SIMPLE_PUSH SCENARIO
# =============================================================================
# Pushing scenario with multiple agents

echo "Starting Simple Push Experiments..."

# Standard training
echo "Training standard MADDPG on simple_push..."
python ../train_new.py --scenario simple_push --exp-name kaRun1 --num-episodes 30000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for kaRun1best..."
python ../train_new.py --scenario simple_push --exp-name kaRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./kaRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for kaRun1..."
python ../train_new.py --scenario simple_push --exp-name kaRun1 --mode train_diffusion --diffusion-data-path ./kaRun1_diffusion_data.npz --diffusion-model-path ./kaRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test with diffusion
echo "Testing kaRun1best with diffusion model..."
python ../train_new.py --scenario simple_push --exp-name kaRun1best --mode test --diffusion-model-path ./kaRun1_diffusion_model.pt

# ERNIE-enhanced training
# echo "Training ERNIE-enhanced MADDPG on simple_push..."
# python ../train_new.py --scenario simple_push --exp-name kaeRun1 --num-episodes 60000 --save-rate 500 --use_ernie

# # Collect diffusion data for ERNIE
# echo "Collecting diffusion data for kaeRun1best..."
# python ../train_new.py --scenario simple_push --exp-name kaeRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./kaeRun1_diffusion_data.npz

# # Train diffusion for ERNIE
# echo "Training diffusion model for kaeRun1..."
# python ../train_new.py --scenario simple_push --exp-name kaeRun1 --mode train_diffusion --diffusion-data-path ./kaeRun1_diffusion_data.npz --diffusion-model-path ./kaeRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# # Test ERNIE with diffusion
# echo "Testing kaeRun1best with diffusion model..."
# python ../train_new.py --scenario simple_push --exp-name kaeRun1best --mode test --diffusion-model-path ./kaeRun1_diffusion_model.pt

echo "Simple Push experiments completed!"