#!/bin/bash

# Script to run the train.py program with varying beta and gama values
# and save the outputs to individual files.

# Path to save the output files
#OUTPUT_DIR="/home/code/result/"
OUTPUT_DIR="./tmp"
timestamp=$(date +"%Y%m%d_%H%M%S")
OUTPUT_SUBDIR="${OUTPUT_DIR}/output_${timestamp}"
mkdir -p "${OUTPUT_SUBDIR}"

# fixed parameter
lr=0.005
smooth_window_size=7
pre_train_iters=150
dataset='pavia'
device_index=0
lnp=20
outp=100
alpha=3
gama=0
beta=7


# Define the output file with dynamic naming based on parameters
OUTPUT_FILE="${OUTPUT_SUBDIR}/_output_alpha_${alpha}_beta_${beta}_gama_${gama}.txt"

# Display the training configuration
echo "Running Model.py with dataset=${dataset} alpha=${alpha}, beta=${beta}, and gama=${gama}"

source C:/Users/thoma/Documents/School/Master/MasterProef/codebase/SCDSC/venv/Scripts/activate

# Run the Python script with arguments and redirect output to a file
python Model.py \
    --device_index ${device_index} \
    --dataset ${dataset} \
    --lr ${lr} \
    --smooth_window_size ${smooth_window_size} \
    --pre_train_iters ${pre_train_iters} \
    --alpha ${alpha} \
    --beta ${beta} \
    --gama ${gama} | tee $OUTPUT_FILE


# Confirm where the output has been saved
echo "Output saved to $OUTPUT_FILE"
