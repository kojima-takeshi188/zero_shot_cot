import json
import requests
import argparse

from generate_demos_utils import generate_prompts_from_json

def main(num_samples):
    
   generate_prompts_from_json("all_prompts.json", "demos.json", num_samples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("num_samples", type = int)
    args = parser.parse_args()

    main(args.num_samples)