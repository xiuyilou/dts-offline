import time, json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
import os
sys.path.append(os.path.expanduser("/home/zxu161"))
import re
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
        description="Entropy Guided Decoding Tree"
    )
    parser.add_argument("-e", "--entropy_threshold", type=float, default=2.5,
                        help="Entropy threshold to branch")
    parser.add_argument("-k", "--branch_top_k", type=int, default=3,
                        help="Number of branches to create")
    parser.add_argument("-a", "--max_active_hyps", type=int, default=12,
                        help="Maximum active hypothesis to maintain in consideration")
    parser.add_argument("-m", "--max_new_tokens", type=int, default=32768,
                        help="Maximum tokens that can be generated")
    parser.add_argument("-l", "--length_penalty", type=float, default=0.2,
                        help="Length penalty")
    parser.add_argument("-t", "--temperature", type=float, default=0.6,
                        help="Entropy and sampling temperature")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("-n", "--num_trials", type=int, default=5,
                        help="Number of trials for the repeated experiment")
    
    args = parser.parse_args()
    entropy_threshold = args.entropy_threshold
    branch_top_k = args.branch_top_k
    max_active_hyps = args.max_active_hyps
    max_new_tokens = args.max_new_tokens
    length_penalty = args.length_penalty
    temperature = args.temperature
    seed = args.seed
    num_trials = args.num_trials

    print("===== Parameters =====")
    print(f"entropy_threshold : {entropy_threshold}")
    print(f"branch_top_k      : {branch_top_k}")
    print(f"max_active_hyps   : {max_active_hyps}")
    print(f"max_new_tokens    : {max_new_tokens}")
    print(f"length_penalty    : {length_penalty}")
    print(f"temperature       : {temperature}")
    print(f"seed              : {seed}")
    print(f"number of trials  : {num_trials}")
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
    all_branch_events = []

    for s in seeds:
        kvegdt = KVBatchEGDT(model, tokenizer, seed=s)

        correct = 0
        num_tokens = 0
        branch_events = 0
        all_responses = []
        all_stats = []
        gt_answers =[]
        count = 0
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

            out = kvegdt.generate(
            text,
            entropy_threshold=entropy_threshold,    
            branch_top_k=branch_top_k,           
            max_active_hyps=max_active_hyps,       
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,      
            temperature=temperature,
            stop_strs=None,
            prefer_greedy_when_confident=False         
            )
            print(out["stats"])
            all_responses.append(out['text'])
            all_stats.append(out["stats"])
            print(f'Decoded question {count}')
            count += 1

        # Process each response in the batch
        for i in range(len(all_stats)):

            response = all_responses[i]
            stat = all_stats[i]

            num_new_tokens = stat['generated_len']
            num_branch_events = stat['branch_events']
            
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
                "tokens": num_new_tokens,
            })
            
            num_tokens += num_new_tokens
            branch_events += num_branch_events
        torch.cuda.synchronize()
        t1 = time.time()
        t = t1 - t0
        avg_num_tokens = num_tokens / len(test_examples)
        avg_branch_events = branch_events / len(test_examples)
        accuracy = correct / len(test_examples)
        all_accuracy.append(accuracy)
        all_branch_events.append(avg_branch_events)
        all_time.append(t)
        all_token_num.append(avg_num_tokens)


    final_accuracy = sum(all_accuracy)/num_trials
    final_branch_events = sum(all_branch_events)/num_trials
    final_time = sum(all_time)/num_trials
    final_token_num = sum(all_token_num)/num_trials
    answers.insert(0, {'accuracy' : final_accuracy, "average branch events" : final_branch_events, 'average new tokens': final_token_num, 'time': final_time})
    fname = (
        f"aime25_repeated_decoder_1.5B"
        f"entro{fmt_float(entropy_threshold)}_"
        f"k{branch_top_k}_"
        f"max_active_hyps{max_active_hyps}_"
        f"len_pen{fmt_float(length_penalty)}_"
        f"temp{fmt_float(temperature)}_"
        f"trials{num_trials}_"
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
