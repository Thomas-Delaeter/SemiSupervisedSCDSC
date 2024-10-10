#!/bin/bash

# Script to run the train.py program with varying beta and gama values
# and save the outputs to individual files.

# Path to save the output files
OUTPUT_DIR="/home/xianlli/code/Signal_processing/result/ours/Trento/beta-gama"
timestamp=$(date +"%Y%m%d_%H%M%S")
OUTPUT_SUBDIR="${OUTPUT_DIR}/output_${timestamp}"
mkdir -p "${OUTPUT_SUBDIR}"

# fixed parameter
lr=0.005
smooth_window_size=7
pre_train_iters=50
dataset='trento'
device_index=0
lnp=20
outp=100

# Range and step settings for beta

# Range and step settings for gama
#GAMA_START=0.1
#GAMA_END=1
#GAMA_STEP=0.1
#beta=0
#alpha=3
#for gama in $(seq $GAMA_START $GAMA_STEP $GAMA_END)
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
#        --lnp ${lnp} \
#        --outp ${outp} \
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


# Range and step settings for gama
#BETA_START=1.0
#BETA_END=12.0
#BETA_STEP=1.0
#gama=0.3
#alpha=3
#for beta in $(seq $BETA_START $BETA_STEP $BETA_END)
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
#        --lnp ${lnp} \
#        --outp ${outp} \
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
#

# Range and step settings for gama
#ALPHA_START=1
#ALPHA_END=12
#ALPHA_STEP=1
#beta=0
#gama=0
#
#for alpha in $(seq $ALPHA_START $ALPHA_STEP $ALPHA_END)
#do
#    # Define the output file
#    OUTPUT_FILE="${OUTPUT_SUBDIR}/output_alpha_${alpha}_output_beta_${beta}_gama_${gama}.txt"
#
#    # Run the Python program with the specified arguments and save the output
#    echo "Running train.py with alpha=${alpha} beta=${beta} and gama=${gama}"
#    python3 0312_Houston.py --alpha ${alpha}  --beta ${beta} --gama ${gama} > $OUTPUT_FILE
#
#    # Optionally: Print the result or check the output
#    echo "Output saved to $OUTPUT_FILE"
#done
#echo "All tasks completed."


#alpha=3
#gama=0
#beta=1
#
#H_START=0
#H_STEP=1
#H_END=0
#
#for h in $(seq $H_START $H_STEP $H_END)
#do
#    # Define the output file with dynamic naming based on parameters
#    OUTPUT_FILE="${OUTPUT_SUBDIR}/h_${h}_output_alpha_${alpha}_beta_${beta}_gama_${gama}.txt"
#    # Display the training configuration
#    echo "Running Model.py with dataset=${dataset} hierarchy=${h} alpha=${alpha}, beta=${beta}, and gama=${gama}"
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
#    # Confirm where the output has been saved
#    echo "Output saved to $OUTPUT_FILE"
#done


gama=0
alpha=3
beta=1


  # Define the output file with dynamic naming based on parameters
  OUTPUT_FILE="${OUTPUT_SUBDIR}/output_alpha_${alpha}_beta_${beta}_gama_${gama}.txt"
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

