#!/bin/bash

# Training Script for Simple Tag (Predator-Prey) Scenario
# This script contains commands to run MADDPG experiments on the simple_tag scenario
# using the MADDPG algorithm with optional diffusion models and ERNIE enhancements.

# Environment variables
export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

# =============================================================================
# SIMPLE_TAG SCENARIO
# =============================================================================
# Predator-Prey environment where predators learn to catch prey

echo "Starting Simple Tag (Predator-Prey) Experiments..."

# Standard MADDPG Training
echo "Training standard MADDPG on simple_tag..."
python ../train.py --scenario simple_tag --num-adversaries 3 --exp-name ppRun1 --num-episodes 30000 --save-rate 500

# Collect diffusion data for best model
echo "Collecting diffusion data for ppRun1best..."
python ../train.py --scenario simple_tag --exp-name ppRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ppRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for ppRun1..."
python ../train.py --scenario simple_tag --exp-name ppRun1 --mode train_diffusion --diffusion-data-path ./ppRun1_diffusion_data.npz --diffusion-model-path ./ppRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 1000

# Test with diffusion model
echo "Testing ppRun1best with diffusion model..."
python ../train.py --scenario simple_tag --exp-name ppRun1best --mode test --diffusion-model-path ./ppRun1_diffusion_model.pt

# Second run with different data
echo "Collecting diffusion data for ppRun2best..."
python ../train.py --scenario simple_tag --exp-name ppRun2best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ppRun2_diffusion_data.npz

# Train diffusion on merged data
echo "Training diffusion model on merged data..."
python ../train.py --scenario simple_tag --exp-name ppRun1 --mode train_diffusion --diffusion-data-path ./pp_diffusion_data_merged.npz --diffusion-model-path ./ppMergeRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test merged model
echo "Testing ppeRun1best with merged diffusion model..."
python ../train.py --scenario simple_tag --exp-name ppeRun1best --mode test --diffusion-model-path ./ppMergeRun1_diffusion_model.pt

# ERNIE-enhanced training
echo "Training ERNIE-enhanced MADDPG on simple_tag..."
python ../train.py --scenario simple_tag --num-adversaries 3 --exp-name ppeRun1 --num-episodes 30000 --save-rate 500 --use_ernie

# Collect diffusion data for ERNIE
echo "Collecting diffusion data for ppeRun1best..."
python ../train.py --scenario simple_tag --exp-name ppeRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ppeRun1_diffusion_data.npz

# Train diffusion for ERNIE (using ppRun1 data)
echo "Training diffusion model for ppeRun1..."
python ../train.py --scenario simple_tag --exp-name ppeRun1 --mode train_diffusion --diffusion-data-path ./ppRun1_diffusion_data.npz --diffusion-model-path ./ppeRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test ERNIE with diffusion
echo "Testing ppeRun1best with diffusion model..."
python ../train.py --scenario simple_tag --exp-name ppeRun1best --mode test --diffusion-model-path ./ppeRun1_diffusion_model.pt

echo "Simple Tag experiments completed!" data)
echo "Training diffusion models for ERNIE..."
for seed in "${SEEDS[@]}"; do
    echo "Training diffusion model for ppeRun${seed}..."
    python ../train.py --scenario simple_tag --exp-name ppeRun${seed} --mode train_diffusion --diffusion-data-path ./ppRun${seed}_diffusion_data.npz --diffusion-model-path ./ppeRun${seed}_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500 --seed $seed
done