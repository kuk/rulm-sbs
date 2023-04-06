
import re
import os
import json
from dataclasses import dataclass, fields, asdict
from collections import defaultdict

from tqdm import tqdm as log_progress

import pandas as pd

import openai


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


def gusev_en_alpaca_prompt(record):
    if record.input:
        return f'''Instruction: {record.instruction}
Input: {record.input}
Output: '''
    
    else:
        return f'''Instruction: {record.instruction}

Output: '''


def gusev_ru_alpaca_prompt(record):
    if record.input:
        return f'''Задание: {record.instruction}
Вход: {record.input}
Выход: '''
    
    else:
        return f'''Вопрос: {record.instruction}

Ответ: '''


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
        return f'''{record.instruction} Ввод: "{record.input}"'''
    else:
        return record.instruction


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


def gusev_alpaca_complete(prompt, model, tokenizer):
    input_ids = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        num_beams=3,
        max_length=512,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        temperature=1.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=5
    )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    return decoded[len(prompt):]


def chainyo_alpaca_complete(prompt, model, tokenizer):
    # for some reaso important to pass args via config
    from transformers import GenerationConfig

    input_ids = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=0.2,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
        )
    )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    return decoded[len(prompt):]


def wortega_instruct_rugpt_complete(prompt, model, tokenizer):
    input_ids = tokenizer(
        prompt + '<instructionS>',
        return_tensors='pt'
    ).input_ids.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        min_length=20,
        max_new_tokens=512,
        top_k=50,
        top_p=0.7,
        do_sample=True,  
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        repetition_penalty=1.5,  
        length_penalty=1.2,  
        num_beams=4,
    )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    return decoded[len(prompt):]


######
#
#   EVAL
#
#####


def eval_gusev_en_alpaca(record, model, tokenizer):
    prompt = gusev_en_alpaca_prompt(record)
    output = gusev_alpaca_complete(prompt, model, tokenizer)
    return EvalRecord(record.id, prompt, output)


def eval_gusev_ru_alpaca(record, model, tokenizer):
    prompt = gusev_ru_alpaca_prompt(record)
    output = gusev_alpaca_complete(prompt, model, tokenizer)
    return EvalRecord(record.id, prompt, output)


def eval_chainyo_alpaca(record, model, tokenizer):
    prompt = chainyo_alpaca_prompt(record)
    output = chainyo_alpaca_complete(prompt, model, tokenizer)
    return EvalRecord(record.id, prompt, output)


def eval_wortega_instruct_rugpt(record, model, tokenizer):
    prompt = wortega_instruct_rugpt_prompt(record)
    output = wortega_instruct_rugpt_complete(prompt, model, tokenizer)
    return EvalRecord(record.id, prompt, output)


def eval_en_openai(record, model):
    prompt = openai_en_prompt(record)
    complete = (
        openai_chat_complete
        if model == GPT_35_TURBO
        else openai_complete
    )
    output = complete(prompt, model)
    return EvalRecord(record.id, prompt, output)


def eval_ru_openai(record, model):
    prompt = openai_ru_prompt(record)
    complete = (
        openai_chat_complete
        if model == GPT_35_TURBO
        else openai_complete
    )
    output = complete(prompt, model)
    return EvalRecord(record.id, prompt, output)
