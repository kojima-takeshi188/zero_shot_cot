import argparse
import json
import pandas

METHODS = ["few_shot", "few_shot_cot", "few_shot_uninformative_cot", "zero_shot_and_uninformative_cot", "zero_shot_cot"]
DATASETS = ["multiarith", "gsm8k", "city_equation"]
NUM_SAMPLES = [4, 10, 15, 20]

def main(log_prefix, output_dir):
    for d in DATASETS:
        df = {}
        for m in METHODS:
            df[m] = []
            for n in NUM_SAMPLES:
                file_path = f"{log_prefix}_{n}/{d}_{m}_output.jsonl"
                with open(file_path, "r") as f: 
                    data = json.load(f)
                    df[m].append(data["accuracy"])
        data_frame = pandas.DataFrame(df)
        data_frame.index = NUM_SAMPLES
        data_frame.to_csv(f"{output_dir}/{d}_accuracy_record.csv")

                    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("log_prefix", type = str)
    parser.add_argument("output_dir", type = str)
    args = parser.parse_args()

    main(args.log_prefix, args.output_dir)




