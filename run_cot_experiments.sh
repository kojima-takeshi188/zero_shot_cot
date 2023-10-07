
# Run few-shot experiments -- non-COT ICL, COT, unformative COT

#!/bin/bash

# Define the model and other parameters
model="gpt3.5"
log_dir="./logs"
# limit_dataset_size is the number of samples used for testing; it will use all testing samples if put 0.
limit_dataset_size=5

# Array of methods
methods=("few_shot" "few_shot_cot" "few_shot_uninformative_cot" "zero_shot_and_uninformative_cot" "zero_shot_cot")
#methods=("zero_shot_and_uninformative_cot")

# Array of datasets
datasets=("multiarith" "gsm8k" "city_equation")

# Array of number of examples in prompt
num_prompt=("4" "10" "15" "20")


# Loop over each number of examples in prompt, method and dataset combination
for n in "${num_prompt[@]}"
do
    echo "Running with number of examples in prompt = ${n} on ${limit_dataset_size} testing samples"
    cd dataset/city_name_arithmetic
    python3 main_city_name_coordinates.py ${n}
    cd ../grade-school-math
    python3 generate_demos.py ${n}
    cd ../MultiArith
    python3 generate_demos.py ${n}
    cd ../..
    for method in "${methods[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            echo "Running with method=${method} on dataset=${dataset}"
            python3 main.py --method=${method} --model=${model} --dataset=${dataset} --log_dir=${log_dir}_${n} --limit_dataset_size=${limit_dataset_size}
            echo "Finished with method=${method} on dataset=${dataset}"
        done
    done
done

echo "All tasks completed!"
