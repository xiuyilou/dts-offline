import time, json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
import argparse
import sys
import os
sys.path.append(os.path.expanduser("/home/zxu161"))
import re
import random

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
# from transformers import default_data_collator
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tiktoken
import accelerate
# from vllm import LLM, SamplingParams
# import transformers
from entroTree.prelim.getans import extract_answer_llm, extract_answer_qwq, is_float
from tqdm import tqdm
from entroTree.main.kvbatch_decoder import KVBatchEGDT
import argparse
from pathlib import Path

def fmt_float(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if s else "0"

def main():
    parser = argparse.ArgumentParser(
        description="Repeated Normal Inference on AIME2024"
    )
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("-n", "--num_trials", type=int, default=5,
                        help="Number of trials for the repeated experiment")
    parser.add_argument("-t", "--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    
    args = parser.parse_args()
    seed = args.seed
    num_trials = args.num_trials
    temperature = args.temperature
    print("===== Parameters =====")
    print(f"initial seed              : {seed}")
    print(f"number of trials          : {num_trials}")
    print(f"temperature               : {temperature}")
    print("======================")
    


    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, cache_dir="./scratch", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype="auto",
            token=access_token,
            cache_dir="./scratch",
            trust_remote_code=True
        )
    
    test_examples = load_dataset("math-ai/aime25", split="test")
    test_examples = list(test_examples)
    seeds = [seed + i for i in range(num_trials)]
    answers = []
    all_accuracy = []
    all_time = []
    all_token_num = []
    for s in seeds:

        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        set_seed(s) 
        correct = 0
        num_tokens = 0
        tokens = []
        all_responses = []
        gt_answers =[]
        # Prepare prompts for the batch
        torch.cuda.synchronize()
        t0 = time.time()
        for example in test_examples:
            prompt = example["problem"]
            gt_answers.append(example["answer"])

            #deepseek r1
            tail = r" Please reason step by step, and put your final answer within \boxed{}."
            messages = [
            {"role": "user", "content": prompt + tail}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=32768,
                do_sample=True,     
                temperature=temperature,    
            )
            num_new_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
            tokens.append(num_new_tokens)
            num_tokens += num_new_tokens
            gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            all_responses.append(gen)

        # Process each response in the batch
        for i in range(len(all_responses)):

            response = all_responses[i]
            token_num = tokens[i]
            gt_answer = gt_answers[i]

            llm_answer = extract_answer_qwq(response)
            if is_float(llm_answer):
                llm_answer = llm_answer
            else:
                llm_answer = extract_answer_llm(response)
            print(gt_answer, "||", llm_answer)
            if is_float(gt_answer) and is_float(llm_answer):
                try:
                    accept = ( int(round(float(gt_answer)))==int(round(float(llm_answer))) )
                except OverflowError:
                    accept = False
            else:
                accept = False
            
            if accept:
                correct += 1
            answers.append({
                "question": test_examples[i]["problem"],
                "gt_answer": gt_answer,
                "llm_answer": llm_answer,
                "accept":accept,
                "llm_response": response,
                "tokens": token_num,
            })
        torch.cuda.synchronize()
        t1 = time.time()
        t = t1 - t0
        avg_num_tokens = num_tokens / len(test_examples)
        accuracy = correct / len(test_examples)
        print(f'Average Token Number {avg_num_tokens}, accuracy {accuracy}')
        all_accuracy.append(accuracy)
        all_time.append(t)
        all_token_num.append(avg_num_tokens)
    final_accuracy = sum(all_accuracy)/num_trials
    final_time = sum(all_time)/num_trials
    final_token_num = sum(all_token_num)/num_trials

    answers.insert(0, {'accuracy' : final_accuracy, 'average new tokens': final_token_num, 'time': final_time})
    fname = (
        f"aime25_normal_inference_1.5B"
        f"temp{fmt_float(temperature)}_"
        f"seed{seed}.json"
    )

    out_dir = Path("./entroTree/main/json")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
