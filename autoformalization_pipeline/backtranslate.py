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

log_file = f"api_log/openai_api_calls_backtrans_{datetime.now().strftime('%Y-%m-%d')}.json"
logger = logging.getLogger("openai_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(message)s')  
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def backtranslate(
    client,
    model_name,
    questions,
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
    repeat_times = 5
    translated_sentences = []

    from tqdm import tqdm
    for example in tqdm(questions, desc='Backtranslation'):
        translated = []
        for prompt in example['formal_output']:
            try:
                response = completions(
                        model=model_name,
                        messages= [
                            {"role": 'user',
                            "content": f'Convert the formal statement into natural language:\n```lean\n {prompt} \n```\
                                        Please only return the translation and no analysis.'
                            }
                            ],
                        temperature=1,
                        )
            except Exception as e:
                print(f"API 调用失败: {e}")

            
            translation = response.choices[0].message.content
            translated.append(translation)

            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            total_tokens += response.usage.total_tokens

        example['back_translate'] = translated
        translated_sentences.append(example)

    tokens["Prompt Tokens"] += prompt_tokens
    tokens["Completion Tokens"] += completion_tokens
    tokens["Total Tokens"] += total_tokens

    return translated_sentences, tokens
