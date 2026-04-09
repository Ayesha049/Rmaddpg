#!/bin/bash

# Training Script for Simple Speaker Listener Scenario
# This script contains commands to run MADDPG experiments on the simple_speaker_listener scenario
# using the MADDPG algorithm with optional diffusion models and ERNIE enhancements.

# Environment variables
export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

# =============================================================================
# SIMPLE_SPEAKER_LISTENER SCENARIO
# =============================================================================
# Communication scenario with speaker and listener agents

echo "Starting Simple Speaker Listener Experiments..."

# Standard training
echo "Training standard MADDPG on simple_speaker_listener..."
python ../train_new.py --scenario simple_speaker_listener --exp-name ccRun1 --num-episodes 60000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for ccRun1best..."
python ../train_new.py --scenario simple_speaker_listener --exp-name ccRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ccRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for ccRun1..."
python ../train_new.py --scenario simple_speaker_listener --exp-name ccRun1 --mode train_diffusion --diffusion-data-path ./ccRun1_diffusion_data.npz --diffusion-model-path ./ccRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test with diffusion
echo "Testing ccRun1best with diffusion model..."
python ../train_new.py --scenario simple_speaker_listener --exp-name ccRun1best --mode test --diffusion-model-path ./ccRun1_diffusion_model.pt

# # ERNIE-enhanced training
# echo "Training ERNIE-enhanced MADDPG on simple_speaker_listener..."
# python ../train_new.py --scenario simple_speaker_listener --exp-name cceRun1 --num-episodes 30000 --save-rate 500 --use_ernie

# # Collect diffusion data for ERNIE
# echo "Collecting diffusion data for cceRun1best..."
# python ../train_new.py --scenario simple_speaker_listener --exp-name cceRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./cceRun1_diffusion_data.npz

# # Train diffusion for ERNIE
# echo "Training diffusion model for cceRun1..."
# python ../train_new.py --scenario simple_speaker_listener --exp-name cceRun1 --mode train_diffusion --diffusion-data-path ./cceRun1_diffusion_data.npz --diffusion-model-path ./cceRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# # Test ERNIE with diffusion
# echo "Testing cceRun1best with diffusion model..."
# python ../train_new.py --scenario simple_speaker_listener --exp-name cceRun1best --mode test --diffusion-model-path ./cceRun1_diffusion_model.pt

# # Additional run
# echo "Training ccRun2..."
# python ../train_new.py --scenario simple_speaker_listener --exp-name ccRun2 --num-episodes 60000 --save-rate 500

# # Parallel testing
# echo "Running parallel tests for ccRun1..."
# python ../train_new.py --scenario simple_speaker_listener --exp-name ccRun1 --mode test & python ../train_new.py --scenario simple_speaker_listener --exp-name ccRun1best --mode test

echo "Simple Speaker Listener experiments completed!"