import argparse
import json
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as numpy
from transformers import AutoTokenizer
import backoff
import openai
from openai import APIError

import logging
from datetime import datetime
import time

log_file = f"api_log/openai_api_calls_nli_{datetime.now().strftime('%Y-%m-%d')}.json"
logger = logging.getLogger("openai_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(message)s')  
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def get_query(origin, back):
        prompt = "Please check whether the following two math problems is same or different in their mathematical essence:" + \
                'Problem 1:\n' + origin + '\nProblem 2:\n' + back + \
             "Please consider each statement in two problems, they are different if any condition or any goal is different. Return in the following format:\n" + \
                '''{"Same": true/false, "Analysis": "Summarize their consistency and difference in brief"}'''
        return prompt


def nli(
    client,
    model_name,
    questions,
    answer_file,
    tokens,
    max_new_token,
    temperature
):
    
    # backoff for retry
    @backoff.on_exception(
        backoff.expo,
        APIError,
        max_tries=5,
        on_backoff=lambda details: logger.info(json.dumps({
            "event": "backoff",
            "wait": details["wait"],
            "tries": details["tries"],
            "exception": str(details["exception"]),
            "timestamp": datetime.now().isoformat()
        }))
    )
    def completions(model: str, **kwargs):
        start_time = time.time()
        log_entry = {
            "event": "openai_call",
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "request": kwargs,
        }

        try:
            response = client.chat.completions.create(
                model=model,
                **kwargs
            )

            duration = time.time() - start_time
            message = response.choices[0].message.content.strip()
            usage = dict(response.usage) if hasattr(response, 'usage') else None

            response_meta = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "usage": usage,
                "message": message
            }

            log_entry.update({
                "response": response_meta,
                "duration_seconds": round(duration, 4),
                "status": "success"
            })
            logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            log_entry.update({
                "error": str(e),
                "duration_seconds": round(duration, 4),
                "status": "error"
            })
            logger.error(json.dumps(log_entry, ensure_ascii=False, default=str))
            raise

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    translated_pass_status = []
    error_info = []

    
    pass_num = 0
    final_set = []
    for example in tqdm(questions, desc='NLI'):
        same_statement_pass = 0
        statement_result = []
        problem_origin = example['natural_language']
        length = len(example['back_translate'])
        for idx in range(length):
            probelm_back = example['back_translate'][idx]
            prompt = get_query(problem_origin, probelm_back)
            
            try:
                response = completions(
                        model=model_name,
                        messages=[
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        temperature=1,
                        )
            except Exception as e:
                print(f"API 调用失败: {e}")
                continue

            NLI = response.choices[0].message.content
            NLI = NLI.replace("\\", "\\\\")

            try:
                NLI = json.loads(NLI) 
                statement_result.append(NLI)

                if NLI['Same']==True:
                    same_statement_pass += 1
                    if final_set and final_set[-1]['natural_language'] == problem_origin:
                        final_set[-1]['formal_statement'].append(example['formal_output'][idx])
                    else:
                        final_set.append(
                            {
                                'natural_language': problem_origin,
                                'formal_statement': [example['formal_output'][idx]]
                            }
                        )
                else:
                    error_info.append(
                        {
                            'natural_language': problem_origin,
                            'formal_statement': example['formal_output'][idx],
                            'translation_error': NLI['Analysis']
                        }
                    )

            except json.decoder.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                error_info.append(
                        {
                            'natural_language': problem_origin,
                            'formal_statement': example['formal_output'][idx],
                            'translation_error': 'json.decoder.JSONDecodeError'
                        }
                    )
                pass

            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            total_tokens += response.usage.total_tokens
        
        
        if same_statement_pass > 0:
            pass_num += 1
       
            # make sure there is no repetition in correct statement and error_info
            while error_info:
                if error_info[-1]['natural_language'] == problem_origin:
                    error_info.pop()
                else:
                    break
        

        example['nli_output'] = statement_result

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")

    tokens["Prompt Tokens"] += prompt_tokens
    tokens["Completion Tokens"] += completion_tokens
    tokens["Total Tokens"] += total_tokens

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    answer_root = os.path.dirname(answer_file)
    output_file = os.path.join(answer_root, 'final_dataset.json')
    with open(output_file, 'a', encoding='utf-8') as f:
        for item in final_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return error_info, tokens, pass_num