#!/bin/bash

# Training Script for Simple Crypto Scenario
# This script contains commands to run MADDPG experiments on the simple_crypto scenario
# using the MADDPG algorithm with optional diffusion models and ERNIE enhancements.

# Environment variables
export SUPPRESS_MA_PROMPT=1
export CUDA_VISIBLE_DEVICES=""

# =============================================================================
# SIMPLE_CRYPTO SCENARIO
# =============================================================================
# Cryptography scenario with sender and receiver agents

echo "Starting Simple Crypto Experiments..."

# Standard training
echo "Training standard MADDPG on simple_crypto..."
python ../train_new.py --scenario simple_crypto --exp-name ccnRun1 --num-episodes 60000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for ccnRun1best..."
python ../train_new.py --scenario simple_crypto --exp-name ccnRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ccnRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for ccnRun1..."
python ../train_new.py --scenario simple_crypto --exp-name ccnRun1 --mode train_diffusion --diffusion-data-path ./ccnRun1_diffusion_data.npz --diffusion-model-path ./ccnRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test with diffusion
echo "Testing ccnRun1best with diffusion model..."
python ../train_new.py --scenario simple_crypto --exp-name ccnRun1best --mode test --diffusion-model-path ./ccnRun1_diffusion_model.pt

# # ERNIE-enhanced training
# echo "Training ERNIE-enhanced MADDPG on simple_crypto..."
# python ../train_new.py --scenario simple_crypto --exp-name ccneRun1 --num-episodes 60000 --save-rate 500 --use_ernie

# # Collect diffusion data for ERNIE
# echo "Collecting diffusion data for ccneRun1best..."
# python ../train_new.py --scenario simple_crypto --exp-name ccneRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ccneRun1_diffusion_data.npz

# # Train diffusion for ERNIE (using ccnRun1 data)
# echo "Training diffusion model for ccneRun1..."
# python ../train_new.py --scenario simple_crypto --exp-name ccneRun1 --mode train_diffusion --diffusion-data-path ./ccnRun1_diffusion_data.npz --diffusion-model-path ./ccneRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 1000

# # Test ERNIE with diffusion
# echo "Testing ccneRun1best with diffusion model..."
# python ../train_new.py --scenario simple_crypto --exp-name ccneRun1best --mode test --diffusion-model-path ./ccneRun1_diffusion_model.pt

echo "Simple Crypto experiments completed!"