import os
import time
import json
import ctypes
import resource
import tempfile
import traceback
import threading
import subprocess
import multiprocessing as mp
from pprint import pprint

import numpy as np
import datetime

from prover.lean.ast_parser import lean4_parser
from prover.workers import ProcessScheduler
from prover.utils import AttrDict

from collections import defaultdict


DEFAULT_LAKE_PATH = '/home1/xjx/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = '../mathlib4/'


def verify_lean4_file(code, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, last_env=None, verbose=False, timeout=300, allTactics=False, ast=False, premises=False, tactics=False):
    command = dict(cmd=code, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
    if last_env is not None:
        command.update(env=last_env)
    message_str = json.dumps(command, ensure_ascii=False)
    if verbose:
        print(message_str)
    start_time = time.time()
    system_messages = ''
    try:
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            outputs = subprocess.run([lake_path, "exe", 'repl'], stdin=temp_file, capture_output=True, text=True, cwd=lean_workspace, timeout=timeout)
        result = json.loads(outputs.stdout)
        ast_results = lean4_parser(code, result['ast']) if 'ast' in result and result['ast'] else {}
        result = {
            "sorries" : result.get('sorries', []), 
            "tactics" : result.get('tactics', []),
            "errors" : [m for m in result.get('messages', []) if m['severity'] == 'error'],
            "warnings" : [m for m in result.get('messages', []) if m['severity'] == 'warning'],
            "infos" : [m for m in result.get('messages', []) if m['severity'] == 'info'],
            "system_messages" : system_messages,
            "system_errors" : None,
            "ast" : ast_results,
            "verified_code" : code,
        }
        result['pass'] = not result['errors']
        result['complete'] = result['pass'] and not result['sorries'] and not any("declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] for warning in result['warnings'])
    except:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages
        }
    result['verify_time'] = time.time() - start_time
    return result


class Lean4ServerProcess(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, extra_args=AttrDict()):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.extra_args = extra_args

        self.timeout = extra_args.get('timeout', 300)
        self.memory_limit = extra_args.get('memory_limit', -1)
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)
    
    def run(self):
        if self.memory_limit > 0:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit * (1000 ** 3), self.memory_limit * (1000 ** 3))
            )
        while True:
            inputs = self.task_queue.get()
            if inputs is None: # Terminate when receiving None
                break
            for _, request_id, task in inputs:
                if isinstance(task, str):
                    task = dict(code=task)
                if 'timeout' not in task:
                    task['timeout'] = self.timeout
                result = verify_lean4_file(**task)
                if len(result['system_messages']) > 0:
                    retry_start_time = time.time()
                    while ('lean::exception: failed to create thread' in result['system_messages'] or
                           'std::bad_alloc: std::bad_alloc' in result['system_messages'] or
                           'Cannot allocate memory' in result['system_messages']) \
                          and time.time() - retry_start_time < self.timeout:
                        time.sleep(0.1)
                        result = verify_lean4_file(**task)
                with self.lock:
                    self.request_statuses[request_id] = result
                    self.last_output_time.value = time.time()
                    self.complete_count.value += 1


class Lean4ServerScheduler(ProcessScheduler):
    def __init__(self, max_concurrent_requests=64, timeout=300, memory_limit=-1, name='verifier'):
        super().__init__(batch_size=1, name=name)
        
        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args=AttrDict(
                    timeout=timeout,
                    memory_limit=memory_limit,
                )
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f'Complete launching {len(self.processes)} LeanServerProcesses')

        self.timeout = timeout
        self._running_monitor = mp.Value(ctypes.c_bool, True)
        self._last_complete_count = mp.Value(ctypes.c_int, 0)
        self._monitor_process = mp.Process(target=self._monitor)
        self._monitor_process.start()
    
    def _monitor(self):
        while self._running_monitor.value:
            time.sleep(1.0)
            subprocess.run(['killall', 'repl', f'--older-than={int(self.timeout) + 10}s'], capture_output=True)
    
    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        self._running_monitor.value = False
        self._monitor_process.join()
        print(f'All {len(self.processes)} LeanServerProcesses stopped')



def save_chunk(data_chunk, chunk_id, save_path):
    """保存 detailed_output 分块"""
    filename = f"{save_path}_failed_{chunk_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in data_chunk:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def process_data(data, lean4_scheduler, header, batch_size, max_retries=3, save_path="output"):
    # 初始化数据结构
    result_dict = defaultdict(list)  # {natural_language: [passed_formals]}
    pending_queue = []                # (natural_language, formal_str, retry_count)
    detailed_output = []              # 记录所有失败且不需要重试的项
    chunk_id = 0                      # 文件分块编号
    
    # 初始化队列（只处理 natural_language 和 formal_output）
    for sample in data:
        nl = sample['natural_language']
        for formal in sample['formal_output']:
            pending_queue.append( (nl, formal, 0) )


    # 处理队列
    while pending_queue:
        batch = []
        # 取当前批次
        while len(batch) < batch_size and pending_queue:
            batch.append(pending_queue.pop(0))
        
        # 生成验证请求
        submit_list = []
        for nl, formal, _ in batch:
            code = header + formal
            submit_list.append(dict(code=code, ast=True, tactics=True))
        
        # 提交验证
        request_id_list = lean4_scheduler.submit_all_request(submit_list)
        outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
        

        # 处理结果
        for (nl, formal, retry_count), output in zip(batch, outputs_list):

            if output['pass']:
                # 通过：加入结果字典
                result_dict[nl].append(formal)
            else:
                # 未通过：根据错误类型处理
                if output.get('system_errors') is not None and retry_count < max_retries:
                    # 系统错误：尝试重试
                    # if retry_count < max_retries:
                    pending_queue.append( (nl, formal, retry_count + 1) )
                else:
                    # 非系统错误：无成功案例的记录到 detailed_output
                    if not result_dict[nl]: 
                        if output.get('system_errors') is None:
                            detailed_output.append({
                                'natural_language': nl,
                                'formal_output': formal,
                                'error': output['errors']
                            })
                        else:
                            detailed_output.append({
                                'natural_language': nl,
                                'formal_output': formal,
                                'error': 'system_error'
                            })

        # 定期保存 detailed_output（每500条）
        while len(detailed_output) >= 500:
            save_chunk(detailed_output[:500], chunk_id, save_path)
            detailed_output = detailed_output[500:]
            chunk_id += 1
    

    # 保存剩余的 detailed_output
    if len(detailed_output) > 0:
        save_chunk(detailed_output, chunk_id, save_path)
        
    return result_dict, detailed_output


def verify(data, save_root, batch_size=10, concurrent_requests=32, max_retry=10):
    my_header = open('./final/new_header.lean').read()

    os.makedirs(save_root, exist_ok=True)
    
    failed_list = []
    detailed_output = []
    traceback_list = []
    filtered_list = []

    pass_num = 0
    start_idx = 0

    start_time = time.time()
    my_lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=concurrent_requests, timeout=300, memory_limit=20, name='verifier')

    result_dict, failed_list = process_data(
        data=data,
        lean4_scheduler=my_lean4_scheduler,
        header=my_header,
        batch_size=batch_size,
        max_retries=max_retry,
        save_path=os.path.join(save_root, 'compiling_test')
    )
    
    my_lean4_scheduler.close()

    return result_dict, failed_list

