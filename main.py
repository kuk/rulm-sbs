
import os
import json

from tqdm import tqdm as log_progress

import pandas as pd

import openai


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


######
#
#   PROMPT
#
######


def user_oriented_openai_prompt(item):
    if item['input']:
        return '''Задание: {instruction}
Ввод: {input}

Ответ: '''.format_map(item)
    else:
        return '''Задание: {instruction}
        
Ответ: '''.format_map(item)


def user_oriented_ru_alcapa_prompt(item):
    if item['input']:
        return '''Задание: {instruction}
Вход: {input}
Выход: '''.format_map(item)
    
    else:
        return '''Вопрос: {instruction}


Выход: '''.format_map(item)


########
#
#   OPENAI
#
######


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


######
#
#  RU ALPACA
#
####


def ru_alpaca_complete(prompt, model, tokenizer):
    input_ids = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        num_beams=3,
        max_length=256,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        temperature=1.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4
    )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return decoded[len(prompt):]
