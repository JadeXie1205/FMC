import argparse
import json
import os
import re
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as numpy
from transformers import AutoTokenizer
import backoff
import openai
from openai import APIError
import time
from datetime import timedelta

from translate import translate
from backtranslate import backtranslate
from nli import nli


# nohup python main.py --question-file data.json --answer-file log > output.log

################################### deepseek-r1 ##########################################

api_key = "your_api_key"
base_url = "your_base_url"
model_name = 'your_model_name'

# OpenAI
client = openai.OpenAI(api_key=api_key, base_url=base_url)
print(f'model_name: {model_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=None,
        help="The output answer file.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    os.environ["WORLD_SIZE"] = '1'
    os.environ["RANK"] = "0"

    with open(args.question_file, 'r') as f:
        questions = [json.loads(item) for item in f.readlines()]

    
    print('##################' + str(torch.cuda.is_available()))
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    print(f"Output to {args.answer_file}")
    print(f"Num Questions: {len(questions)}")


    batch_size=20

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    answer_root = os.path.dirname(args.answer_file)
    save_root = answer_root + '/'
    save_path = save_root + f'filtered_trans.json'

    first_pass = 0
    second_pass = 0
    after_nli_pass = 0

    first_nli_pass_num = 0
    second_nli_pass_num = 0


    token_num = {
            "Prompt Tokens": 0,
            "Completion Tokens": 0,
            "Total Tokens": 0
            }

    all_start_time = time.time()
    for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"): 
        print()
        print(f"################################## the {i // batch_size + 1} batch ##########################################")
        start_time = time.time()
        data = questions[i:i+batch_size]
        # translation + Lean verification
        passed_result = []
        passed_result, failed_result, token_num = translate(
                                                    client,
                                                    model_name,
                                                    data,
                                                    None,
                                                    None,
                                                    answer_root,
                                                    5, # repeat_times
                                                    token_num,
                                                    args.max_new_token,
                                                    args.temperature, 
                                                    batch_size,
                                                )
        # 1st error feedback (from verification)
        newly_passed = []
        if failed_result:
            newly_passed, failed_result, token_num = translate(
                                                        client,
                                                        model_name,
                                                        None,
                                                        failed_result,
                                                        passed_result,
                                                        answer_root,
                                                        1, # repeat_times
                                                        token_num,
                                                        args.max_new_token,
                                                        args.temperature, 
                                                        batch_size,
                                                    )
        
        filtered_list = [
                            {
                                'natural_language': nl,
                                'formal_output': formal_list
                            } for nl, formal_list in passed_result.items() if len(formal_list) > 0
                        ]
        first_len = len(filtered_list)
        first_pass += first_len
        print(f'{i//batch_size+1} batch: {first_len} passed at first round, ', end="")
        if newly_passed:
            filtered_list.extend(
                [
                    {
                        'natural_language': nl,
                        'formal_output': formal_list
                    } for nl, formal_list in newly_passed.items() if len(formal_list) > 0
                ]
            )
        second_len = len(filtered_list)-first_len
        second_pass += second_len
        print(f'{second_len} passed at the second round. Total {first_len + second_len} passing compiling.')


        # backtranslation
        if filtered_list:
            save_root = answer_root + '/'
            save_path = save_root + f'filtered_trans.json'
            for item in filtered_list:
                with open(save_path, "a") as fout:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            # backtranslation + consistency check
            backtransed, token_num = backtranslate(client, model_name, filtered_list, token_num,
                                        args.max_new_token,
                                        args.temperature)

            save_root = answer_root + '/'
            answer_file = save_root + 'nli_result.json'
            nli_error, token_num, first_nli_pass = nli(client, model_name, backtransed, answer_file, token_num,
                                        args.max_new_token,
                                        args.temperature) 
            first_nli_pass_num += first_nli_pass
            
            # 2nd error feedback (from consistency check)
            second_passed = []
            if nli_error:
                second_passed, failed_result, token_num = translate(
                                                            client,
                                                            model_name,
                                                            None,
                                                            nli_error,
                                                            [],
                                                            answer_root,
                                                            1, # repeat_times
                                                            token_num,
                                                            args.max_new_token,
                                                            args.temperature,
                                                            batch_size,
                                                        )

            new_filtered_list = []
            if second_passed:
                new_filtered_list = [
                        {
                            'natural_language': nl,
                            'formal_output': formal_list
                        } for nl, formal_list in second_passed.items() if len(formal_list) > 0
                    ]

                after_nli_pass += len(new_filtered_list)
                print(f'{len(new_filtered_list)} passed compiling after nli revision.')

            if new_filtered_list:
                save_root = answer_root + '/'
                save_path = save_root + f'filtered_trans.json'
                for item in new_filtered_list:
                    with open(save_path, "a") as fout:
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
                new_backtransed, token_num = backtranslate(client, model_name, new_filtered_list, token_num, 
                                                                                args.max_new_token,
                                                                                args.temperature)
     
                nli_error, token_num, second_nli_pass  = nli(client, model_name, new_backtransed, answer_file, token_num, 
                                                                        args.max_new_token,
                                                                        args.temperature)
            else:
                second_nli_pass = 0
            
            second_nli_pass_num += second_nli_pass

        end_time = time.time()
        duration = end_time - start_time

        # convert to hour:minute:second format
        formatted_duration = str(timedelta(seconds=round(duration)))
        print()
        print(f"the {i // batch_size + 1} batch using time: ", formatted_duration)

        print(f"Compiling pass: first_pass = {first_pass}, second_pass = {second_pass}")
        print(f"NLI pass: first_pass = {first_nli_pass_num}, second_pass = {second_nli_pass_num}")
        print(f"tokens: {token_num}")
        print()


    all_end_time = time.time()
    all_duration = all_end_time - all_start_time

    print('####################################### ALL TRANSLATION FINISHED #######################################')

    all_formatted_duration = str(timedelta(seconds=round(all_duration)))
    print(f"Total using time: ", all_formatted_duration)
    print(f"Compiling pass: first_pass = {first_pass}, second_pass = {second_pass}")
    print(f"NLI pass: first_pass = {first_nli_pass_num}, second_pass = {second_nli_pass_num}")
    print(f"tokens: {token_num}")
    print(f'Saved in {save_root}' + 'final_dataset.json.')
                        
