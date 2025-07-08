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

from verifier_deepseek_fun import verify

import logging
from datetime import datetime
import time

log_file = f"api_log/openai_api_calls_trans_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
logger = logging.getLogger("openai_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(message)s') 
    fh.setFormatter(formatter)
    logger.addHandler(fh)




few_shot='''Natural language version: "Simplify $\left( \frac{4}{x} \right)^{-1} \left( \frac{3x^3}{x} \right)^2 \left( \frac{1}{2x} \right)^{-3}$. The final answer is 18x^8"\n
Translate the natural language version to a Lean 4 version:\n
theorem test\n
  (x : ℝ)\n
  (h₀ : x ≠ 0) :\n
  (4 / x)⁻¹ * ((3 * x^3) / x)^2 * ((1 / (2 * x))⁻¹)^3 = 18 * x^8 := by sorry\n


Natural language version: "If $n$ is a positive integer such that $2n$ has 28 positive divisors and $3n$ has 30 positive divisors, then how many positive divisors does $6n$ have? The final answer is 35"\n
Translate the natural language version to a Lean version:\n
theorem test\n
  (n : ℕ)\n
  (h₀ : 0 < n)\n
  (h₁ : Finset.card (Nat.divisors (2 * n)) = 28)\n
  (h₂ : Finset.card (Nat.divisors (3 * n)) = 30) :\n
  Finset.card (Nat.divisors (6 * n)) = 35 := by sorry\n
'''


def get_query(problem, query, few_shot=None, failed_info=None):
        if few_shot:
            prompt = query+ f'Here are some examples for it: {few_shot}\n' + \
                    f'Following the examples above, translate the next problem into Lean4: {problem}'
        if failed_info:
            prompt = query + f'Here is the theorem in natural language: {problem}\n' + \
                    f'Before your translation, note that this problem has been mistranslated as the following. Concrete errors have been listed and \
                        please avoid similar mistakes when translating it again. Mistranslation: {failed_info}'
        return prompt


def translate(
        client,
        model_name,
        questions,
        failed_info,
        result_dict,
        answer_root,
        repeat_times,
        tokens,
        max_new_token,
        temperature,
        batch_size,
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
    translated_sentences = []

    error_file = os.path.join(answer_root, 'translate_error.json')


    if failed_info and not questions:
        '''
            {
                'natural_language': nl,
                'formal_output': formal,
                'error': output['errors'] / 'translation_error'
            }
        '''
        questions = []
        failed = []
        for item in failed_info:
            if len(result_dict)!= 0 and result_dict[item['natural_language']] != []:
                continue
            else:
                questions.append({'natural_language': item['natural_language']})
                failed.append(item)


    from tqdm import tqdm
    for idx, example in enumerate(tqdm(questions, desc='Translation')):
        problem = str(example['natural_language'])
        query = 'A math theorem in natural language will be provided and please translate it into a Lean4 theorem. \
                Please only return the translation (Lean4 code) and no analysis, no mathlib4 import, no comments, no proof, no reasoning.\
                Use ":= by sorry" as a placeholder for proof.\n'
        if failed_info:
            prompt = get_query(problem, query, failed_info=failed[idx])
        else:
            prompt = get_query(problem, query, few_shot=few_shot)

        translated = []
        for i in range(repeat_times):
            try:
                response = completions(
                        model=model_name,
                        messages= [{"role": 'user', "content": prompt}],
                        temperature=1,
                        )
                translation = response.choices[0].message.contents
                pattern = r'(theorem\b(?:.(?![Ll]ean4))*?by sorry)'
                match = re.search(pattern, translation, re.DOTALL)

                if match:
                    translated.append(match.group(1)) 
                else:
                    error = {'natural_language': example['natural_language'],
                            'formal_error': translation}
                    with open(error_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(error, ensure_ascii=False) + "\n")

                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens
                total_tokens += response.usage.total_tokens
            except Exception as e:
                print(f"API 调用失败: {e}")

        translated_sentences.append({'natural_language': example['natural_language'],
                                    'formal_output': translated})

    
    tokens["Prompt Tokens"] += prompt_tokens
    tokens["Completion Tokens"] += completion_tokens
    tokens["Total Tokens"] += total_tokens


    save_root = answer_root + '/'
    result_dict, failed_info = verify(translated_sentences, save_root, batch_size=batch_size)

    return result_dict, failed_info, tokens