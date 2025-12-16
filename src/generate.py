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
    
class Review(BaseModel):
    explanation: str
    is_self_contained: bool


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
    
CHECK_SELF_CONTAINED_PROMPT = """You are analyzing a user message from a training example.
The upstream generator that produced this example is lazy and often leaves user messages with missing or implicit context.
Your job is to classify the user message into self-contained and not self-contained categories.

If the user message misses context, it is not self-contained.
Example:
The message "I have this paragraph in a textbook and I need help summarizing it. Can you do that?" is NOT self-contained because it asks about the paragraph but it doesn't provide the paragraph above or below.
The message "I have this paragraph in a textbook and I need help summarizing it. Can you do that?" asks for help summarizing a paragraph from a textbook. However, it doesn't provide the text itself. Therefore, it’s not self-contained.

# The messages (JSON):

{example_json}

# Output format:
{{
    "explanation": "...",
    "is_self_contained": true or false,
}}

# Begin review:
"""

def check_self_contained(example):
    ex_json = example.model_dump_json(indent=4)
    prompt = CHECK_SELF_CONTAINED_PROMPT.format(example_json=ex_json)
    resp = llm_check.invoke(prompt)
    assert isinstance(resp, Review)
    print(resp)
    return resp.is_self_contained, resp.explanation

def dummy_gen(comb, n_samples):
    examples: list[Example] = []
    for i in range(n_samples):
        examples.append(Example(system_message="system message ...", user_message="user message ..."))
    return Examples(examples=examples)

def generate_examples(comb, n_examples, explanation_str):
    explanation_str = "\n=== SELF-CONTAINED MESSAGE ===\n" + explanation_str + "\n"
    topic, intent, strength, style = comb
    prompt = GEN_PROMPT.format(
        constraint_strength_description=STRENGTH_WORDINGS[strength],
        topic_description=TOPIC_WORDINGS[topic],
        user_intent_description=INTENT_WORDINGS[intent],
        style_description=STYLE_WORDINGS[style],
        n_examples=n_examples,
        check_self_contained_explanation=explanation_str
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
    explanations = []
    n_todo = n_samples
    while n_todo > 0:
        explanation_str = ""
        for ex, explanation in explanations:
            explanation_str += f"The message \"{ex.user_message}\" is not self-contained. {explanation}"
            
        examples = generate_examples(comb, min(5, n_todo), explanation_str)
        n_accepted = 0
        for ex in examples:
            is_self_contained, explanation = check_self_contained(ex)
            if is_self_contained:
                all_examples.append(ex)
                n_accepted += 1
            else:
                if len(explanations) < 5:
                    explanations.append((ex, explanation))
        n_todo -= n_accepted


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
    
    global llm, llm_check
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"), # type: ignore
        api_key=os.getenv("API_KEY"), # type: ignore
        base_url=os.getenv("BASE_URL"),
    )
    llm_check = llm.with_structured_output(Review)
    llm = llm.with_structured_output(Examples)

    for i in range(start_idx, end_idx):
        work(df, i)