
import re
import os
import json
import random
import asyncio
from pathlib import Path
from dataclasses import dataclass, fields, asdict
from collections import defaultdict

from tqdm import tqdm as log_progress

import pandas as pd

import openai

# LLM.int8() requires Turing or Ampere GPUs.
# WARNING: No libcudart.so found! Install CUDA or the cudatoolkit package (anaconda)!

import torch
from peft import (
    PeftModel,
    PeftConfig
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,

    GPT2TokenizerFast,
    GPT2LMHeadModel,

    GenerationConfig
)


GPT_35_TURBO = 'gpt-3.5-turbo'
TEXT_DACINCI_003 = 'text-davinci-003'
TEXT_DACINCI_002 = 'text-davinci-002'


#######
#
#   LINES
#
######


def read_lines(path):
    with open(path) as file:
        for line in file:
            yield line.rstrip('\n')


def write_lines(path, lines):
    with open(path, 'w') as file:
        for line in lines:
            file.write(line + '\n')


#######
#
#   JSONL
#
#####


def parse_jsonl(lines):
    for line in lines:
        yield json.loads(line)


def format_jsonl(items):
    for item in items:
        yield json.dumps(item, ensure_ascii=False, indent=None)


########
#
#   DOTENV
#
######


DOTENV_PATH = '.env'


def parse_dotenv(lines):
    for line in lines:
        if line:
            key, value = line.split('=', 1)
            yield key, value


####
#
#   OBJ
#
#####


@dataclass
class TaskRecord:
    id: str
    category: str = None
    instruction: str = None
    input: str = None


@dataclass
class EvalRecord:
    id: str
    prompt: str
    output: str


def item_records(items, cls):
    for item in items:
        yield cls(**item)


def record_items(records):
    for record in records:
        yield asdict(record)


def load_task(path):
    lines = read_lines(path)
    items = parse_jsonl(lines)
    return item_records(items, TaskRecord)


def load_eval(path):
    lines = read_lines(path)
    items = parse_jsonl(lines)
    return item_records(items, EvalRecord)


def dump_eval(path, records):
    items = record_items(records)
    lines = format_jsonl(items)
    write_lines(path, lines)


######
#
#   PROMPT
#
######


def openai_en_prompt(record):
    if record.input:
        return f'''Instruction: {record.instruction}
Input: {record.input}

Output: '''
    else:
        return record.instruction


def openai_ru_prompt(record):
    if record.input:
        return f'''Задание: {record.instruction}
Ввод: {record.input}

Ответ: '''
    else:
        return record.instruction


def gusev_alpaca_prompt(record):
    if record.input:
        return f'''Instruction: {record.instruction}
Input: {record.input}
Output: '''
    
    else:
        return f'''Instruction: {record.instruction}

Output: '''


def gusev_ru_alpaca_prompt(record):
    if record.input:
        return f'''### Задание: {record.instruction}
### Вход: {record.input}
### Ответ: '''
    
    else:
        return f'''### Задание: {record.instruction}
### Ответ: '''


def gusev_saiga_prompt(record):
    content = record.instruction
    if record.input:
        content = f'''{content}

{record.input}
'''

    return f'''<start>system
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. <end><start>user
{content} <end>
<start>system
'''


def chainyo_alpaca_prompt(record):
    if record.input:
        return f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{record.instruction}

### Input:
{record.input}

### Response:'''
    else:
        return f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{record.instruction}

### Response:'''


def wortega_instruct_rugpt_prompt(record):
    if record.input:
        return f'''{record.instruction} Ввод: "{record.input}"<instructionS>'''
    else:
        return f'{record.instruction}<instructionS>'


######
#
#    COMPLETE
#
####


def openai_complete(prompt, model='text-davinci-003'):
    completion = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=1024,
    )
    return completion.choices[0].text


def openai_chat_complete(prompt, model='gpt-3.5-turbo'):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1024,
    )
    return completion.choices[0].message.content


def model_batch_complete(records, model, tokenizer, generation_config, batch_size=8):
    records = [_ for _ in records if not _.output]
    records = sorted(
        records,
        key=lambda _: len(_.prompt)
    )

    for index in range(0, len(records), batch_size):
        batch_records = records[index:index + batch_size]
        batch = tokenizer(
            [_.prompt for _ in batch_records],
            return_tensors='pt',
            padding=True
        ).to(model.device)

        outputs = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            generation_config=generation_config
        )

        for record, output in zip(batch_records, outputs):
            output = tokenizer.decode(
                output,
                skip_special_tokens=True
            )
            record.output = output[len(record.prompt):]
            print(record)


#######
#
#   LLAMACPP
#
########


async def llamacpp_complete(prompt, main_path, model_path, threads, generation_args):
    proc = await asyncio.create_subprocess_exec(
        str(main_path),
        '--prompt', prompt,
        '--model', str(model_path),
        '--threads', str(threads),
        *generation_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    data = await proc.stdout.read()
    output = data.decode('utf8')
    return output[len(prompt):]


async def batch_llamacpp_complete(
        records, main_path, model_path, generation_args,
        threads=8, pool_size=4
):
    records = [_ for _ in records if not _.output]
    records = iter(records)
    
    async def worker(records):
        for record in records:
            record.output = await llamacpp_complete(
                record.prompt,
                main_path, model_path,
                threads, generation_args
            )
            print(record)

    tasks = [
        worker(records)
        for _ in range(pool_size)
    ]
    await asyncio.gather(*tasks)
