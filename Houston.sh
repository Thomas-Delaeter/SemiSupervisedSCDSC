#!/bin/bash

# Script to run the train.py program with varying beta and gama values
# and save the outputs to individual files.

# Path to save the output files
#OUTPUT_DIR="/home/xianlli/code/Signal_processing/result/ours/Houston/beta-gama"
OUTPUT_DIR="./tmp"
timestamp=$(date +"%Y%m%d_%H%M%S")
OUTPUT_SUBDIR="${OUTPUT_DIR}/output_${timestamp}"
mkdir -p "${OUTPUT_SUBDIR}"

# fixed parameter
lr=0.0001
smooth_window_size=3
pre_train_iters=10
dataset='Houston'
device_index=0
lnp=20
outp=100


gama=0
alpha=3
beta=8

# Define the output file with dynamic naming based on parameters
OUTPUT_FILE="${OUTPUT_SUBDIR}/output_alpha_${alpha}_beta_${beta}_gama_${gama}.txt"

source C:/Users/thoma/Documents/School/Master/MasterProef/codebase/SCDSC/venv/Scripts/activate

# Display the training configuration
echo "Running Model.py with lnp=${lnp} outp=${outp} alpha=${alpha}, beta=${beta}, and gama=${gama}"

# Run the Python script with arguments and redirect output to a file
python Model.py \
    --device_index ${device_index} \
    --dataset ${dataset} \
    --lr ${lr} \
    --smooth_window_size ${smooth_window_size} \
    --pre_train_iters ${pre_train_iters} \
    --alpha ${alpha} \
    --lnp ${lnp} \
    --outp ${outp} \
    --beta ${beta} \
    --gama ${gama} > $OUTPUT_FILE
# Confirm where the output has been saved
echo "Output saved to $OUTPUT_FILE"

