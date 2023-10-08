#!/bin/bash
# Run few-shot experiments -- non-COT ICL, COT, unformative COT

# Define the model and other parameters
model="gpt4"
log_dir="./logs_gpt4"
# limit_dataset_size is the number of samples used for testing; it will use all testing samples if put 0.
limit_dataset_size=100

# Array of methods
#methods=("few_shot" "few_shot_cot" "few_shot_uninformative_cot" "uninformative_cot_with_trigger" "zero_shot_cot")

#methods=("uninformative_cot_with_trigger")
#methods=("uninformative_demographics_cot_with_trigger")
#methods=("ICL_with_trigger")
#methods=("few_shot_cot")

#methods=("few_shot")
methods=("few_shot_cot" "uninformative_cot_with_trigger" "uninformative_demographics_cot_with_trigger" "ICL_with_trigger")


# Array of datasets
#datasets=("multiarith" "gsm8k" "city_equation")
datasets=("city_equation")

# Array of number of examples in prompt
num_prompt=("10")


# Loop over each number of examples in prompt, method and dataset combination
for n in "${num_prompt[@]}"
do
    echo "Running with number of examples in prompt = ${n} on ${limit_dataset_size} testing samples"
    #cd dataset/city_name_arithmetic
    #/usr/bin/python3 main_city_name_coordinates.py ${n}
    #cd ../grade-school-math
    #/usr/bin/python3 generate_demos.py ${n}
    #cd ../MultiArith
    #/usr/bin/python3 generate_demos.py ${n}
    #cd ../..
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
