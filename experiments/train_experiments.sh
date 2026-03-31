#!/bin/bash

# Training Script for Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Experiments
# This script contains commands to run various experiments across different scenarios
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
python train.py --scenario simple_tag --num-adversaries 3 --exp-name ppRun1 --num-episodes 30000 --save-rate 500

# Collect diffusion data for best model
echo "Collecting diffusion data for ppRun1best..."
python train.py --scenario simple_tag --exp-name ppRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ppRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for ppRun1..."
python train.py --scenario simple_tag --exp-name ppRun1 --mode train_diffusion --diffusion-data-path ./ppRun1_diffusion_data.npz --diffusion-model-path ./ppRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 1000

# Test with diffusion model
echo "Testing ppRun1best with diffusion model..."
python train.py --scenario simple_tag --exp-name ppRun1best --mode test --diffusion-model-path ./ppRun1_diffusion_model.pt

# Second run with different data
echo "Collecting diffusion data for ppRun2best..."
python train.py --scenario simple_tag --exp-name ppRun2best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ppRun2_diffusion_data.npz

# Train diffusion on merged data
echo "Training diffusion model on merged data..."
python train.py --scenario simple_tag --exp-name ppRun1 --mode train_diffusion --diffusion-data-path ./pp_diffusion_data_merged.npz --diffusion-model-path ./ppMergeRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test merged model
echo "Testing ppeRun1best with merged diffusion model..."
python train.py --scenario simple_tag --exp-name ppeRun1best --mode test --diffusion-model-path ./ppMergeRun1_diffusion_model.pt

# ERNIE-enhanced training
echo "Training ERNIE-enhanced MADDPG on simple_tag..."
python train.py --scenario simple_tag --num-adversaries 3 --exp-name ppeRun1 --num-episodes 30000 --save-rate 500 --use_ernie

# Collect diffusion data for ERNIE
echo "Collecting diffusion data for ppeRun1best..."
python train.py --scenario simple_tag --exp-name ppeRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ppeRun1_diffusion_data.npz

# Train diffusion for ERNIE (using ppRun1 data)
echo "Training diffusion model for ppeRun1..."
python train.py --scenario simple_tag --exp-name ppeRun1 --mode train_diffusion --diffusion-data-path ./ppRun1_diffusion_data.npz --diffusion-model-path ./ppeRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test ERNIE with diffusion
echo "Testing ppeRun1best with diffusion model..."
python train.py --scenario simple_tag --exp-name ppeRun1best --mode test --diffusion-model-path ./ppeRun1_diffusion_model.pt

# =============================================================================
# SIMPLE_ADVERSARY SCENARIO
# =============================================================================
# Two-agent environment with one adversary and one good agent

echo "Starting Simple Adversary Experiments..."

# Standard training
echo "Training standard MADDPG on simple_adversary..."
python train.py --scenario simple_adversary --exp-name saRun1 --num-episodes 60000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for saRun1best..."
python train.py --scenario simple_adversary --exp-name saRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./saRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for saRun1..."
python train.py --scenario simple_adversary --exp-name saRun1 --mode train_diffusion --diffusion-data-path ./saRun1_diffusion_data.npz --diffusion-model-path ./saRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 1000

# Test with diffusion
echo "Testing saRun1best with diffusion model..."
python train.py --scenario simple_adversary --exp-name saRun1best --mode test --diffusion-model-path ./saRun1_diffusion_model.pt

# ERNIE-enhanced training
echo "Training ERNIE-enhanced MADDPG on simple_adversary..."
python train.py --scenario simple_adversary --exp-name pdeRun1 --num-episodes 60000 --save-rate 500 --use_ernie

# Collect diffusion data for ERNIE
echo "Collecting diffusion data for pdeRun1best..."
python train.py --scenario simple_adversary --exp-name pdeRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./pdeRun1_diffusion_data.npz

# Train diffusion for ERNIE
echo "Training diffusion model for pdeRun1..."
python train.py --scenario simple_adversary --exp-name pdeRun1 --mode train_diffusion --diffusion-data-path ./pdeRun1_diffusion_data.npz --diffusion-model-path ./pdeRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test ERNIE with diffusion
echo "Testing pdeRun1best with diffusion model..."
python train.py --scenario simple_adversary --exp-name pdeRun1best --mode test --diffusion-model-path ./pdeRun1_diffusion_model.pt

# Additional runs
echo "Training pdSRRun2..."
python train.py --scenario simple_adversary --exp-name pdSRRun2 --num-episodes 30000 --save-rate 500

# Parallel testing
echo "Running parallel tests for pdSRRun1..."
python train.py --scenario simple_adversary --exp-name pdSRRun1 --mode test & python train.py --scenario simple_adversary --exp-name pdSRRun1best --mode test

# =============================================================================
# SIMPLE_SPEAKER_LISTENER SCENARIO
# =============================================================================
# Communication scenario with speaker and listener agents

echo "Starting Simple Speaker Listener Experiments..."

# Standard training
echo "Training standard MADDPG on simple_speaker_listener..."
python train.py --scenario simple_speaker_listener --exp-name ccRun1 --num-episodes 30000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for ccRun1best..."
python train.py --scenario simple_speaker_listener --exp-name ccRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ccRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for ccRun1..."
python train.py --scenario simple_speaker_listener --exp-name ccRun1 --mode train_diffusion --diffusion-data-path ./ccRun1_diffusion_data.npz --diffusion-model-path ./ccRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 1000

# Test with diffusion
echo "Testing ccRun1best with diffusion model..."
python train.py --scenario simple_speaker_listener --exp-name ccRun1best --mode test --diffusion-model-path ./ccRun1_diffusion_model.pt

# ERNIE-enhanced training
echo "Training ERNIE-enhanced MADDPG on simple_speaker_listener..."
python train.py --scenario simple_speaker_listener --exp-name cceRun1 --num-episodes 30000 --save-rate 500 --use_ernie

# Collect diffusion data for ERNIE
echo "Collecting diffusion data for cceRun1best..."
python train.py --scenario simple_speaker_listener --exp-name cceRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./cceRun1_diffusion_data.npz

# Train diffusion for ERNIE
echo "Training diffusion model for cceRun1..."
python train.py --scenario simple_speaker_listener --exp-name cceRun1 --mode train_diffusion --diffusion-data-path ./cceRun1_diffusion_data.npz --diffusion-model-path ./cceRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test ERNIE with diffusion
echo "Testing cceRun1best with diffusion model..."
python train.py --scenario simple_speaker_listener --exp-name cceRun1best --mode test --diffusion-model-path ./cceRun1_diffusion_model.pt

# Additional run
echo "Training ccRun2..."
python train.py --scenario simple_speaker_listener --exp-name ccRun2 --num-episodes 60000 --save-rate 500

# Parallel testing
echo "Running parallel tests for ccRun1..."
python train.py --scenario simple_speaker_listener --exp-name ccRun1 --mode test & python train.py --scenario simple_speaker_listener --exp-name ccRun1best --mode test

# =============================================================================
# SIMPLE_CRYPTO SCENARIO
# =============================================================================
# Cryptography scenario with sender and receiver agents

echo "Starting Simple Crypto Experiments..."

# Standard training
echo "Training standard MADDPG on simple_crypto..."
python train.py --scenario simple_crypto --exp-name ccnRun1 --num-episodes 30000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for ccnRun1best..."
python train.py --scenario simple_crypto --exp-name ccnRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ccnRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for ccnRun1..."
python train.py --scenario simple_crypto --exp-name ccnRun1 --mode train_diffusion --diffusion-data-path ./ccnRun1_diffusion_data.npz --diffusion-model-path ./ccnRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 1000

# Test with diffusion
echo "Testing ccnRun1best with diffusion model..."
python train.py --scenario simple_crypto --exp-name ccnRun1best --mode test --diffusion-model-path ./ccnRun1_diffusion_model.pt

# ERNIE-enhanced training
echo "Training ERNIE-enhanced MADDPG on simple_crypto..."
python train.py --scenario simple_crypto --exp-name ccneRun1 --num-episodes 60000 --save-rate 500 --use_ernie

# Collect diffusion data for ERNIE
echo "Collecting diffusion data for ccneRun1best..."
python train.py --scenario simple_crypto --exp-name ccneRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./ccneRun1_diffusion_data.npz

# Train diffusion for ERNIE (using ccnRun1 data)
echo "Training diffusion model for ccneRun1..."
python train.py --scenario simple_crypto --exp-name ccneRun1 --mode train_diffusion --diffusion-data-path ./ccnRun1_diffusion_data.npz --diffusion-model-path ./ccneRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 1000

# Test ERNIE with diffusion
echo "Testing ccneRun1best with diffusion model..."
python train.py --scenario simple_crypto --exp-name ccneRun1best --mode test --diffusion-model-path ./ccneRun1_diffusion_model.pt

# =============================================================================
# SIMPLE_PUSH SCENARIO
# =============================================================================
# Pushing scenario with multiple agents

echo "Starting Simple Push Experiments..."

# Standard training
echo "Training standard MADDPG on simple_push..."
python train.py --scenario simple_push --exp-name kaRun1 --num-episodes 30000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for kaRun1best..."
python train.py --scenario simple_push --exp-name kaRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./kaRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for kaRun1..."
python train.py --scenario simple_push --exp-name kaRun1 --mode train_diffusion --diffusion-data-path ./kaRun1_diffusion_data.npz --diffusion-model-path ./kaRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 1000

# Test with diffusion
echo "Testing kaRun1best with diffusion model..."
python train.py --scenario simple_push --exp-name kaRun1best --mode test --diffusion-model-path ./kaRun1_diffusion_model.pt

# ERNIE-enhanced training
echo "Training ERNIE-enhanced MADDPG on simple_push..."
python train.py --scenario simple_push --exp-name kaeRun1 --num-episodes 60000 --save-rate 500 --use_ernie

# Collect diffusion data for ERNIE
echo "Collecting diffusion data for kaeRun1best..."
python train.py --scenario simple_push --exp-name kaeRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./kaeRun1_diffusion_data.npz

# Train diffusion for ERNIE
echo "Training diffusion model for kaeRun1..."
python train.py --scenario simple_push --exp-name kaeRun1 --mode train_diffusion --diffusion-data-path ./kaeRun1_diffusion_data.npz --diffusion-model-path ./kaeRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test ERNIE with diffusion
echo "Testing kaeRun1best with diffusion model..."
python train.py --scenario simple_push --exp-name kaeRun1best --mode test --diffusion-model-path ./kaeRun1_diffusion_model.pt

# =============================================================================
# SIMPLE_SPREAD SCENARIO
# =============================================================================
# Cooperative spreading scenario

echo "Starting Simple Spread Experiments..."

# Standard training
echo "Training standard MADDPG on simple_spread..."
python train.py --scenario simple_spread --exp-name cnRun1 --num-episodes 60000 --save-rate 500

# Collect diffusion data
echo "Collecting diffusion data for cnRun1best..."
python train.py --scenario simple_spread --exp-name cnRun1best --mode collect_diffusion --num-episodes 2000 --diffusion-horizon 25 --diffusion-data-path ./cnRun1_diffusion_data.npz

# Train diffusion model
echo "Training diffusion model for cnRun1..."
python train.py --scenario simple_spread --exp-name cnRun1 --mode train_diffusion --diffusion-data-path ./cnRun1_diffusion_data.npz --diffusion-model-path ./cnRun1_diffusion_model.pt --diffusion-horizon 25 --diffusion-steps 100 --diffusion-epochs 500

# Test with diffusion
echo "Testing cnRun1best with diffusion model..."
python train.py --scenario simple_spread --exp-name cnRun1best --mode test --diffusion-model-path ./cnRun1_diffusion_model.pt

# Test another model
echo "Testing cneRun1best with diffusion model..."
python train.py --scenario simple_spread --exp-name cneRun1best --mode test --diffusion-model-path ./cneRun1_diffusion_model.pt

# Additional training
echo "Training cnSRRun1..."
python train.py --scenario simple_spread --exp-name cnSRRun1 --num-episodes 60000 --save-rate 500

# Parallel testing
echo "Running parallel tests for cnSRRun1..."
python train.py --scenario simple_spread --exp-name cnSRRun1 --mode test & python train.py --scenario simple_spread --exp-name cnSRRun1best --mode test

# =============================================================================
# MISCELLANEOUS COMMANDS
# =============================================================================

echo "Running miscellaneous test commands..."

# Test ppRun4best
python train.py --scenario simple_tag --exp-name ppRun4best --mode test

# Short ERNIE training runs
python train.py --scenario simple_tag --num-adversaries 3 --exp-name ppeRun1 --num-episodes 6000 --save-rate 500 --use_ernie
python train.py --scenario simple_speaker_listener --exp-name ccRun1 --num-episodes 60000 --save-rate 500 --use_ernie

echo "All experiments completed!"