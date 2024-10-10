#!/bin/bash

# Script to run the train.py program with varying beta and gama values
# and save the outputs to individual files.

# Path to save the output files
OUTPUT_DIR="/home/xianlli/code/Signal_processing/result/ours/PaviaU/beta-gama"
timestamp=$(date +"%Y%m%d_%H%M%S")
OUTPUT_SUBDIR="${OUTPUT_DIR}/output_${timestamp}"
mkdir -p "${OUTPUT_SUBDIR}"

# fixed parameter
lr=0.005
smooth_window_size=7
pre_train_iters=150
dataset='pavia'
device_index=1
lnp=20
outp=100

# Range and step settings for beta
#BETA_START=0
#BETA_END=0
#BETA_STEP=0

# Range and step settings for gama
#GAMA_START=0.0
#GAMA_END=1.0
#GAMA_STEP=0.1
#beta=0
#alpha=3
#
#for gama in $(seq $GAMA_START $GAMA_STEP $GAMA_END)
#do
#    # Define the output file with dynamic naming based on parameters
#    OUTPUT_FILE="${OUTPUT_SUBDIR}/output_beta_${beta}_gama_${gama}.txt"
#
#    # Display the training configuration
#    echo "Running Model.py with dataset=${dataset} alpha=${alpha}, beta=${beta}, and gama=${gama}"
#
#    # Run the Python script with arguments and redirect output to a file
#    python3 Model.py \
#        --device_index ${device_index} \
#        --dataset ${dataset} \
#        --lr ${lr} \
#        --smooth_window_size ${smooth_window_size} \
#        --pre_train_iters ${pre_train_iters} \
#        --alpha ${alpha} \
#        --lnp ${lnp} \
#        --outp ${outp} \
#        --beta ${beta} \
#        --gama ${gama} > $OUTPUT_FILE
#
#
#    # Confirm where the output has been saved
#    echo "Output saved to $OUTPUT_FILE"
#done


#Alpha_START=1
#Alpha_END=13
#ALPHA_STEP=1
#beta=0
#gama=0
#
#for alpha in $(seq $Alpha_START $ALPHA_STEP $Alpha_END)
#do
#    # Define the output file with dynamic naming based on parameters
#    OUTPUT_FILE="${OUTPUT_SUBDIR}/output_alpha_${alpha}_beta_${beta}_gama_${gama}.txt"
#
#    # Display the training configuration
#    echo "Running Model.py with dataset=${dataset} alpha=${alpha}, beta=${beta}, and gama=${gama}"
#
#    # Run the Python script with arguments and redirect output to a file
#    python3 Model.py \
#        --device_index ${device_index} \
#        --dataset ${dataset} \
#        --lr ${lr} \
#        --smooth_window_size ${smooth_window_size} \
#        --pre_train_iters ${pre_train_iters} \
#        --alpha ${alpha} \
#        --beta ${beta} \
#        --gama ${gama} > $OUTPUT_FILE
#
#
#    # Confirm where the output has been saved
#    echo "Output saved to $OUTPUT_FILE"
#done



#BETAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
#Beta_START=1
#Beta_END=12
#BETA_STEP=1
#alpha=3
#gama=0
#
#for beta in $(seq $Beta_START $BETA_STEP $Beta_END)
#do
#    # Define the output file with dynamic naming based on parameters
#    OUTPUT_FILE="${OUTPUT_SUBDIR}/output_alpha_${alpha}_beta_${beta}_gama_${gama}.txt"
#
#    # Display the training configuration
#    echo "Running Model.py with dataset=${dataset} alpha=${alpha}, beta=${beta}, and gama=${gama}"
#
#    # Run the Python script with arguments and redirect output to a file
#    python3 Model.py \
#        --device_index ${device_index} \
#        --dataset ${dataset} \
#        --lr ${lr} \
#        --smooth_window_size ${smooth_window_size} \
#        --pre_train_iters ${pre_train_iters} \
#        --alpha ${alpha} \
#        --beta ${beta} \
#        --gama ${gama} > $OUTPUT_FILE
#
#
#    # Confirm where the output has been saved
#    echo "Output saved to $OUTPUT_FILE"
#done



#alpha=3
#gama=0
#beta=0
#
#H_START=0
#H_STEP=1
#H_END=0
#
#for h in $(seq $H_START $H_STEP $H_END)
#do
#    # Define the output file with dynamic naming based on parameters
#    OUTPUT_FILE="${OUTPUT_SUBDIR}/h_${h}_output_alpha_${alpha}_beta_${beta}_gama_${gama}.txt"
#
#    # Display the training configuration
#    echo "Running Model.py with dataset=${dataset} hierarchy=${h} alpha=${alpha}, beta=${beta}, and gama=${gama}"
#
#    # Run the Python script with arguments and redirect output to a file
#    python3 Model.py \
#        --device_index ${device_index} \
#        --dataset ${dataset} \
#        --lr ${lr} \
#        --hierarchy ${h} \
#        --smooth_window_size ${smooth_window_size} \
#        --pre_train_iters ${pre_train_iters} \
#        --alpha ${alpha} \
#        --beta ${beta} \
#        --gama ${gama} > $OUTPUT_FILE
#
#
#    # Confirm where the output has been saved
#    echo "Output saved to $OUTPUT_FILE"
#done


alpha=3
gama=0
beta=7


# Define the output file with dynamic naming based on parameters
OUTPUT_FILE="${OUTPUT_SUBDIR}/_output_alpha_${alpha}_beta_${beta}_gama_${gama}.txt"

# Display the training configuration
echo "Running Model.py with dataset=${dataset} alpha=${alpha}, beta=${beta}, and gama=${gama}"

# Run the Python script with arguments and redirect output to a file
python3 Model.py \
    --device_index ${device_index} \
    --dataset ${dataset} \
    --lr ${lr} \
    --smooth_window_size ${smooth_window_size} \
    --pre_train_iters ${pre_train_iters} \
    --alpha ${alpha} \
    --beta ${beta} \
    --gama ${gama} > $OUTPUT_FILE


# Confirm where the output has been saved
echo "Output saved to $OUTPUT_FILE"
