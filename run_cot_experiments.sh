
# Run few-shot experiments -- non-COT ICL, COT, unformative COT

#!/bin/bash

# Define the model and other parameters
model="gpt3.5"
log_dir="./logs"
limit_dataset_size=2

# Array of methods
methods=("few_shot" "few_shot_cot" "few_shot_uninformative_cot")

# methods=("few_shot_uninformative_cot")
# Array of datasets
#datasets=("multiarith" "gsm8k", "city_equation")
# datasets=("city_equation")
datasets=("multiarith" "gsm8k" "city_equation")
# Loop over each method and dataset combination
for method in "${methods[@]}"
do
    for dataset in "${datasets[@]}"
    do
        echo "Running with method=${method} on dataset=${dataset}"
        python3 main.py --method=${method} --model=${model} --dataset=${dataset} --log_dir=${log_dir} --limit_dataset_size=${limit_dataset_size}
        echo "Finished with method=${method} on dataset=${dataset}"
    done
done

echo "All tasks completed!"
