
# Run few-shot experiments -- non-COT ICL, COT, unformative COT

#!/bin/bash

# Define the model and other parameters
model="gpt3-xl"
log_dir="/log/fewshot_cot/"
limit_dataset_size=0

# Array of methods
methods=("few_shot" "few_shot_cot" "few_shot_uninformative_cot", "few_shot_uninformative_cot")

# methods=("few_shot_uninformative_cot")
# Array of datasets
datasets=("multiarith" "gsm8k")

# Loop over each method and dataset combination
for method in "${methods[@]}"
do
    for dataset in "${datasets[@]}"
    do
        echo "Running with method=${method} on dataset=${dataset}"
        python main.py --method=${method} --model=${model} --dataset=${dataset} --log_dir=${log_dir} --limit_dataset_size=${limit_dataset_size}
        echo "Finished with method=${method} on dataset=${dataset}"
    done
done

echo "All tasks completed!"
