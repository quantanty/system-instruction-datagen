import argparse
import json
import os
import re
from typing import List

import pandas as pd
from dotenv import load_dotenv
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel

from src.prompt_divergence import (
    GEN_PROMPT,
    INTENT_WORDINGS,
    STRENGTH_WORDINGS,
    STYLE_WORDINGS,
    TOPIC_WORDINGS,
)

load_dotenv()

tag: str = ""

class Example(BaseModel):
    system_message: str
    user_message: str

class Examples(BaseModel):
    examples: List[Example]


def save(file, example, topic, intent, strength, style):
    obj = {
        "system_message": example.system_message,
        "user_message": example.user_message,
        "meta": {
            "topic": topic,
            "intent": intent,
            "strength": strength,
            "style": style,
        }
    }
    line = json.dumps(obj) + "\n"
    file.write(line)

def dummy_gen(comb, n_samples):
    examples: list[Example] = []
    for i in range(n_samples):
        examples.append(Example(system_message="system message ...", user_message="user message ..."))
    return Examples(examples=examples)

def generate_examples(comb, n_examples):
    topic, intent, strength, style = comb
    prompt = GEN_PROMPT.format(
        constraint_strength_description=STRENGTH_WORDINGS[strength],
        topic_description=TOPIC_WORDINGS[topic],
        user_intent_description=INTENT_WORDINGS[intent],
        style_description=STYLE_WORDINGS[style],
        n_examples=n_examples
    )
    assert llm
    resp = llm.invoke(prompt)
    return resp.examples # type: ignore

def work(df, idx):
    row = df.iloc[idx]
    topic = row.topic
    intent = row.intent
    strength = row.strength
    style = row.style
    n_samples = row.n_samples

    comb = (topic, intent, strength, style)
    
    all_examples = []
    n_todo = n_samples
    while n_todo > 0:
        examples = generate_examples(comb, min(5, n_todo))
        n_todo -= len(examples)
        all_examples += examples


    global tag
    if tag:
        tag_postfix = "-" + tag
    else:
        tag = ""
    os.makedirs("outputs/combinations/", exist_ok=True)
    with open(f"outputs/combinations/{idx}{tag_postfix}.jsonl", "a") as f:
        for i, ex in enumerate(all_examples):
            save(f, ex, topic, intent, strength, style)
            # print(f"Ex {i}.\nSystem Message:\n{ex.system_message}\n"
            #     f"User Message:\n{ex.user_message}")
    print(f"Done combination {comb}, n todo: {n_samples}, n generated: {len(all_examples)}, file: {f.name}")
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-file",
        type=str,
        help="Path to CSV file that containts combinations and number of samples for each combination."
    )
    parser.add_argument(
        "--rows",
        type=str,
        default="[:]",
        help="row indices, e.g., [4:10]. Default: [:]"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Add tag in file names. Default: \"\""
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    sample_file = args.sample_file
    regex = r"\[(\d*):(\d*)\]"
    rows_str = args.rows
    
    match = re.search(regex, rows_str)
    if match:
        start_idx_str = match.group(1)
        end_idx_str = match.group(2)
        
        start_idx = int(start_idx_str) if start_idx_str else 0
        end_idx = int(end_idx_str) if end_idx_str else None
    else:
        raise ValueError(f"Invalid slice: {rows_str}")

    df = pd.read_csv(sample_file)
    if start_idx >= len(df):
        raise ValueError(f"Invalid slice: {rows_str}, '{sample_file}' only has {len(df)} data rows")
    if not end_idx:
        end_idx = len(df)
    end_idx = min(end_idx, len(df))
    
    print(args.sample_file)
    print(start_idx, end_idx)
    print(args.tag)
    tag = args.tag
    
    global llm
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"), # type: ignore
        api_key=os.getenv("API_KEY"), # type: ignore
        base_url=os.getenv("BASE_URL"),
    ).with_structured_output(Examples)

    for i in range(start_idx, end_idx):
        work(df, i)