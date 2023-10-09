import argparse
import logging
import torch
import random
import time
import os
from utils import *
import logging


def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY"))
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    
    # if args.method == "few_shot":
    #     demo = create_demo_text(args, cot_flag=False)
    # elif args.method == "few_shot_cot":
    #     demo = create_demo_text(args, cot_flag=True)
    # else:
    #     pass
    
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_type="non-cot")
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_type="informative-cot")
    elif args.method == "few_shot_cot_with_trigger":
        demo = create_demo_text(args, cot_type="informative-cot")
    elif args.method == "few_shot_uninformative_cot":  # you'll need to add support for this in your argument parser
        demo = create_demo_text(args, cot_type="uninformative-cot")
    elif args.method == "uninformative_cot_with_trigger":  
        demo = create_demo_text(args, cot_type="uninformative-cot-with-trigger")
    elif args.method == "ICL_with_trigger":
        demo = create_demo_text(args, cot_type="non-cot")
    elif args.method == "uninformative_demographics_cot_with_trigger":  
        demo = create_demo_text(args, cot_type="uninformative-demographics-cot")
    else:
        pass


    total = 0
    # diff = 0
    correct_list = []

    # Ensure the output directory exists
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    output_file_path = os.path.join(args.log_dir, args.dataset+"_" + args.method+"_output.jsonl")

    # Open the file for appending
    # with open(output_file_path, "w") as wp:  
    output = []   
    for i, data in enumerate(dataloader):

        output_line = {}
        print('*************************')
        print("{}st data".format(i+1))
                
        # Prepare question template ...
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()

        output_line["new_prompt"] = x
        output_line["ground truth"] = y

        print("testing question : " + x)

        if args.method == "zero_shot":
            x = x + " " + args.direct_answer_trigger_for_zeroshot
        elif args.method == "zero_shot_cot":
            #x = x + " " + args.cot_trigger
            x = x + " Concisely explain your steps and write your answer as a integer in the last sentence starting with 'The answer is'. "
        elif args.method == "few_shot":
            x = demo + x
        elif args.method == "few_shot_cot":
            x = demo + x
        elif args.method == "ICL_with_trigger":
            x = demo + x + " Concisely explain your steps and write your answer as a integer in the last sentence starting with 'The answer is'. "
        elif args.method == "few_shot_cot_with_trigger":
            x = demo + x + " Concisely explain your steps and write your answer as a integer in the last sentence starting with 'The answer is'. "
        elif args.method == "few_shot_uninformative_cot":
            x = demo + x
        elif args.method == "uninformative_cot_with_trigger":
            x = demo + x + " Concisely explain your steps and write your answer as a integer in the last sentence starting with 'The answer is'. "
        elif args.method == "uninformative_demographics_cot_with_trigger":
            x = demo + x + " Concisely explain your steps and write your answer as a integer in the last sentence starting with 'The answer is'. "
        else:
            raise ValueError("method is not properly defined ...")

        print("\n the prompt/input given to the language model is \n"+x+"\n")
        
        # Answer prediction by generating text ...
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        try:
            z = decoder.decode(args, x, max_length, i, 1)
        except:
            z = ""

        

        # Answer extraction for zero-shot-cot ...
        if args.method == "zero_shot_cot":
            z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            pred = decoder.decode(args, z2, max_length, i, 2)
            print(z2 + pred)
        else:
            pred = z
            print("the output of language model is: "+pred)
            
        output_line["llm_output"] = pred
        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)
        
        output_line["pred_ans"] = pred
        # output_line["wrap_que"] = x

        
        #output_json = json.dumps(output_line, indent=2)
        #wp.write(output_json + '\n')

        output.append(output_line)
        # Choose the most frequent answer from the list ...
        
        print("pred : {}".format(pred))
        print("GT : " + y)
        #print("type of GT" , type(y))
        print('*************************')
        
        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
        # diff += np.abs(pred - y)
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            #raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))

    # put accuracy to the log file
    final_output = {"accuracy": accuracy, "outputs": output}
    with open(output_file_path, "w") as wp: 
        wp.write(json.dumps(final_output, indent=2))
    
    # total_average_diff = diff / total
    # print("total_average_diff : {}".format(total_average_diff))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters", "city_equation"], help="dataset used for experiment"
    )
    
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt3", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "gpt3-xxl", "gpt3.5", "gpt4"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cot_with_trigger", "few_shot_uninformative_cot", "uninformative_cot_with_trigger","ICL_with_trigger","uninformative_demographics_cot_with_trigger"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=1024, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.few_shot_prompt_path = "./dataset/grade-school-math/demos.json"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.few_shot_prompt_path = "./dataset/MultiArith/demos.json"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "city_equation":
        args.dataset_path = "./dataset/city_name_arithmetic/test_set.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals)is"
        args.few_shot_prompt_path = "./dataset/city_name_arithmetic/demos.json"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    
    args.direct_answer_trigger_for_fewshot = "The answer is"
    
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    return args

if __name__ == "__main__":
    main()