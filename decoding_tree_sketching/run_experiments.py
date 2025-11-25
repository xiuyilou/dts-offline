import time, json
import torch
import numpy as np
import random
import sys
import os
import re
import argparse
from pathlib import Path
import yaml

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,  
    set_seed,
)

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tiktoken
import accelerate

from decoding_tree_sketching.utils.eval_utils import extract_answer_llm, extract_answer_qwq, is_float
from decoding_tree_sketching.kvbatch_decoder import KVBatchEGDT

def fmt_float(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if s else "0"

def extract_answer_gsm8k(text: str):
    """
    Extract numeric final answer from GSM8K format, e.g. '...#### 42' -> '42'.
    """
    if text is None:
        return None
    
    # CORRECTED REGEX: Look for the '####' marker followed by the number
    m = re.search(r'####\s*(\-?[0-9\.\,]+)', text)
    
    if m:
        # Found the marker, extract the number
        match_str = m.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        # If the marker isn't found, the GT format is unexpected
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run DTS or Standard Inference Experiment"
    )
    
    # --- Mode Argument ---
    parser.add_argument(
        "mode", 
        choices=["dts", "standard"],
        help="Execution mode: 'dts' for DTS or 'standard' for normal inference"
    )

    # --- Config Selection Arguments ---
    parser.add_argument("--model_name", type=str, default="1.5B", choices=["1.5B", "7B", "phi-4-mini-reasoning"],
                        help="Model configuration key from configs/config.yaml (e.g., '1.5B')")
    parser.add_argument("--dataset_name", type=str, default="aime24", choices=["aime24", "aime25", "GSM8K", "gpqa_diamond"],
                        help="Dataset configuration key from configs/config.yaml (e.g., 'aime24')")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml",
                        help="Path to the experiment configuration file")

    # --- DTS Arguments ---
    parser.add_argument("-e", "--entropy_threshold", type=float, default=2.5,
                        help="[DTS only] Entropy threshold to branch")
    parser.add_argument("-k", "--branch_top_k", type=int, default=3,
                        help="[DTS only] Number of branches to create")
    parser.add_argument("-a", "--max_active_hyps", type=int, default=12,
                        help="[DTS only] Maximum active hypothesis to maintain")

    # --- Shared Arguments ---
    parser.add_argument("-m", "--max_new_tokens", type=int, default=32768,
                        help="Maximum tokens that can be generated")
    parser.add_argument("-t", "--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Initial random seed")
    parser.add_argument("-n", "--num_trials", type=int, default=5,
                        help="Number of trials for the repeated experiment")
    parser.add_argument("--tag", type=str, default="fast@12",
                        help="Optional tag to append to the output filename (before .json)")

    args = parser.parse_args()

    # --- 1. Load Config File ---
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config_file}")
        sys.exit(1)
    
    # Get specific configs based on args
    try:
        model_config = config["models"][args.model_name]
        dataset_config = config["datasets"][args.dataset_name]
        general_config = config["general"]
        output_config = config["output"]
    except KeyError as e:
        print(f"Error: Config file {args.config_file} is missing required key: {e}")
        sys.exit(1)


    print("===== Parameters =====")
    print(f"mode                : {args.mode}")
    print(f"model_config        : {args.model_name}")
    print(f"dataset_config      : {args.dataset_name}")
    print(f"initial seed        : {args.seed}")
    print(f"number of trials    : {args.num_trials}")
    print(f"temperature         : {args.temperature}")
    print(f"max_new_tokens      : {args.max_new_tokens}")
    if args.mode == "dts":
        print(f"entropy_threshold   : {args.entropy_threshold}")
        print(f"branch_top_k        : {args.branch_top_k}")
        print(f"max_active_hyps     : {args.max_active_hyps}")
    print("======================")

    # --- 2. Load Model and Tokenizer (from Config) ---
    model_name_hf = model_config["model_name"]
    access_token = general_config.get("hf_access_token")
    cache_dir = general_config.get("cache_dir", "./.cache")

    if access_token:
        print("Warning: Found 'hf_access_token' in config. Using env variables is safer.")
        
    print(f"Loading model: {model_name_hf}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_hf, token=access_token, cache_dir=cache_dir, trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_hf,
        device_map="auto",
        torch_dtype="auto",
        token=access_token,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )


    # --- 3. Load Dataset (from Config) ---
    dataset_name_hf = dataset_config["dataset_name"]
    dataset_split = dataset_config.get("split", "train")
    prompt_tail = dataset_config["prompt_tail"]
    prompt_key = dataset_config.get("prompt_key", "Problem") 
    answer_key = dataset_config.get("answer_key", "Answer") 

    print(f"Loading dataset: {dataset_name_hf} (split: {dataset_split})...")
    if dataset_name_hf == "openai/gsm8k":
        test_examples = load_dataset(dataset_name_hf, "main", split=dataset_split)
    elif dataset_name_hf == "Idavidrein/gpqa":
        test_examples = load_dataset(dataset_name_hf, "gpqa_diamond", split=dataset_split)
    else:
        test_examples = load_dataset(dataset_name_hf, split=dataset_split)
    test_examples = list(test_examples)
    
    seeds = [args.seed + i for i in range(args.num_trials)]
    answers = []
    all_accuracy = []
    all_time = []
    all_token_num = []
    all_branch_events = []
    all_detailed_candidates_only = []
    # --- 4. Run Trials Loop ---
    for s in seeds:
        print(f"\n--- Running Trial with Seed {s} ---")
        
        correct = 0
        num_tokens = 0
        branch_events = 0
        all_responses = []
        all_stats = []
        gt_answers =[]
        
        torch.cuda.synchronize()
        t0 = time.time()

        if args.mode == "dts":
            kvegdt = KVBatchEGDT(model, tokenizer, seed=s)
            
            random.seed(s) 
            np.random.seed(s)
            
            target_indices = {0,3,7,16,20,21,6,10,13,17,23,27,8,19,26}
            for index, example in enumerate(test_examples):
                if index in target_indices:
                    
                    prompt_content = ""
                    
                    if args.dataset_name == "gpqa_diamond":
                        question_text = example[prompt_key]
                        correct_opt = example[answer_key]
                        
                        options = [
                            example[answer_key], 
                            example["Incorrect Answer 1"],
                            example["Incorrect Answer 2"],
                            example["Incorrect Answer 3"]
                        ]
                        
                        options = [str(opt).strip() if opt is not None else "" for opt in options]
                        correct_opt_cleaned = str(correct_opt).strip()

                        random.shuffle(options)
                        
                        try:
                            correct_index = options.index(correct_opt_cleaned)
                        except ValueError:
                            print(f"Error: Correct answer '{correct_opt_cleaned}' not found in options list: {options}")
                            continue 
                            
                        correct_letter_gt = ['A', 'B', 'C', 'D'][correct_index]
                        
                        gt_answers.append(correct_letter_gt)
                        
                        formatted_options = [f"{['A', 'B', 'C', 'D'][i]}. {opt}" for i, opt in enumerate(options)]
                        options_str = "\n".join(formatted_options)
                        
                        prompt_content = f"{question_text}\n\n{options_str}\n\n{prompt_tail}"
                    
                    else:
                        prompt_content = example[prompt_key] + prompt_tail
                        gt_answers.append(example[answer_key])

                    messages = [{"role": "user", "content": prompt_content}]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    out = kvegdt.generate(
                        text,
                        entropy_threshold=args.entropy_threshold,    
                        branch_top_k=args.branch_top_k,          
                        max_active_hyps=args.max_active_hyps,      
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        stop_strs=None,
                        prefer_greedy_when_confident=False,
                        num_traces=12,
                    )
                    
                    traces = out.get("traces", [])
                    global_stats = out.get("stats", {})

                    all_token_metrics = global_stats.get("all_token_metrics", [])

                    all_traces_info = []
                    if len(traces) == 0:
                        best_text = ""
                        best_tokens = []
                        best_reward = None
                    else:
                        cand_texts = [tr["text"] for tr in traces]

                        for tr in traces:
                            all_traces_info.append({
                                "text": tr["text"],
                                "token_count": len(tr["tokens"]),
                                "logprob": tr["logprob"],
                                "reward_score": None,
                                "entropy_history": tr.get("entropy_history", []),
                                "varentropy_history": tr.get("varentropy_history", []),
                                "token_metrics": tr.get("token_metrics", []),
                                "branch_steps": tr.get("branch_steps", []),
                            })


                        best_trace = max(traces, key=lambda tr: tr["logprob"])
                        best_text = best_trace["text"]
                        best_tokens = best_trace["tokens"]
                        best_reward = None

                    gen_len = len(best_tokens)
                    branch_ev = global_stats.get("branch_events", 0)
                    
                    num_candidates = len(traces)

                    all_responses.append(best_text)
                    all_stats.append({
                        "generated_len": gen_len,
                        "branch_events": branch_ev,
                        "reward_score": best_reward,
                        "num_candidates": num_candidates,  
                        "total_branches_created": global_stats.get("total_branches_created", None),
                        "all_candidate_traces": all_traces_info,
                        "all_token_metrics": all_token_metrics 
                    })

                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        elif args.mode == "standard":
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            set_seed(s) 
            
            for example in tqdm(test_examples, desc=f"Std. Gen (Seed {s})"):
                
                prompt_content = ""
                
                if args.dataset_name == "gpqa_diamond":
                    question_text = example[prompt_key]
                    correct_opt = example[answer_key]
                    
                    options = [
                        example[answer_key], 
                        example["Incorrect Answer 1"],
                        example["Incorrect Answer 2"],
                        example["Incorrect Answer 3"]
                    ]
                    
                    options = [str(opt).strip() if opt is not None else "" for opt in options]
                    correct_opt_cleaned = str(correct_opt).strip()


                    random.shuffle(options)
                    
                    try:
                        correct_index = options.index(correct_opt_cleaned)
                    except ValueError:
                        print(f"Error: Correct answer '{correct_opt_cleaned}' not found in options list: {options}")
                        continue 
                        
                    correct_letter_gt = ['A', 'B', 'C', 'D'][correct_index]
                    
                    gt_answers.append(correct_letter_gt)
                    
                    formatted_options = [f"{['A', 'B', 'C', 'D'][i]}. {opt}" for i, opt in enumerate(options)]
                    options_str = "\n".join(formatted_options)
                    
                    prompt_content = f"{question_text}\n\n{options_str}\n\n{prompt_tail}"
                
                else:

                    prompt_content = example[prompt_key] + prompt_tail
                    gt_answers.append(example[answer_key])

                messages = [{"role": "user", "content": prompt_content}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,    
                    temperature=args.temperature,    
                )
                
                num_new_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
                gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                all_responses.append(gen)
                all_stats.append({
                    'generated_len': num_new_tokens,
                    'branch_events': 0
                })

        # --- 5. Process Responses (Shared) ---
        sorted_indices = sorted(list(target_indices))
        print("Evaluating responses...")
        for i, real_idx in enumerate(sorted_indices):
            response = all_responses[i]
            stat = all_stats[i]
            
            gt_raw = gt_answers[i]
            if "openai/gsm8k" in dataset_name_hf.lower():
                gt_answer = extract_answer_gsm8k(gt_raw)
            else:
                gt_answer = str(gt_raw) 

            llm_answer = extract_answer_qwq(response)
            if not llm_answer:
                 llm_answer = extract_answer_llm(response)

            accept = False
            
            if args.dataset_name == "gpqa_diamond":
                clean_llm_ans = None
                clean_gt_ans = None
                
                if llm_answer:
                    clean_llm_ans = llm_answer.strip().strip('.').upper()
                if gt_answer:
                    clean_gt_ans = gt_answer.strip().strip('.').upper()
                
                if clean_llm_ans and clean_gt_ans:
                    if clean_gt_ans in ['A', 'B', 'C', 'D'] and clean_llm_ans == clean_gt_ans:
                        accept = True
                
                llm_answer = clean_llm_ans
                gt_answer = clean_gt_ans

            else:
                if is_float(gt_answer) and is_float(llm_answer):
                    try:
                        accept = ( int(round(float(gt_answer))) == int(round(float(llm_answer))) )
                    except (OverflowError, ValueError):
                        accept = False
            
            if accept:
                correct += 1
            
            detailed_candidates = []
            candidate_traces = stat.get("all_candidate_traces", None)
            if candidate_traces:
                for cand in candidate_traces:
                    cand_text = cand["text"]
                    
                    cand_llm_answer = extract_answer_qwq(cand_text)
                    if not cand_llm_answer:
                        cand_llm_answer = extract_answer_llm(cand_text)

                    cand_accept = False
                    cand_gt_answer = gt_answer  

                    if args.dataset_name == "gpqa_diamond":
                        cand_clean_llm = cand_llm_answer.strip().strip('.').upper() if cand_llm_answer else None
                        cand_clean_gt = cand_gt_answer.strip().strip('.').upper() if cand_gt_answer else None

                        if cand_clean_llm and cand_clean_gt:
                            if cand_clean_gt in ['A', 'B', 'C', 'D'] and cand_clean_llm == cand_clean_gt:
                                cand_accept = True

                        cand_llm_answer = cand_clean_llm
                        cand_gt_answer = cand_clean_gt
                    else:
                        if is_float(cand_gt_answer) and is_float(cand_llm_answer):
                            try:
                                cand_accept = (
                                    int(round(float(cand_gt_answer)))
                                    == int(round(float(cand_llm_answer)))
                                )
                            except (OverflowError, ValueError):
                                cand_accept = False

                    detailed_candidates.append({
                        "question": test_examples[real_idx][prompt_key],
                        "gt_answer": cand_gt_answer,
                        "llm_answer": cand_llm_answer,
                        "accept": cand_accept,
                        "llm_response": cand_text,
                        "tokens": cand.get("token_count", None),  
                        "branch_events": len(cand.get("branch_steps", [])),  
                        "entropy_history": cand.get("entropy_history", []),
                        "varentropy_history": cand.get("varentropy_history", []),
                        "token_metrics": cand.get("token_metrics", []),
                        "branch_steps": cand.get("branch_steps", []),
                    })

                slim_candidates = []
                for cand_item in detailed_candidates:
                    slim_candidates.append({
                        "llm_answer": cand_item["llm_answer"],
                        "accept": cand_item["accept"],
                        "llm_response": cand_item["llm_response"],
                        "tokens": cand_item["tokens"],
                        "branch_events": cand_item["branch_events"],
                        "branch_steps": cand_item.get("branch_steps", []),
                    })

                all_detailed_candidates_only.append({
                    "seed": s,                      
                    "index": real_idx+1,                   
                    "question": test_examples[real_idx][prompt_key],
                    "gt_answer": gt_answer,
                    "candidates": slim_candidates,
                })

            answers.append({
                "question": test_examples[real_idx][prompt_key], 
                "gt_answer": gt_answer,
                "llm_answer": llm_answer,
                "accept": accept,
                "llm_response": response, 
                "tokens": stat["generated_len"],
                "branch_events": stat["branch_events"],
                "num_candidates": stat.get("num_candidates", None), 
                "total_branches_created": stat.get("total_branches_created", None),
                "all_candidate_traces": detailed_candidates,
                "all_token_metrics": stat.get("all_token_metrics", []),
            })

            num_tokens += stat['generated_len']
            branch_events += stat['branch_events']
            
        torch.cuda.synchronize()
        t1 = time.time()
        t = t1 - t0
        
        avg_num_tokens = num_tokens / len(test_examples)
        avg_branch_events = branch_events / len(test_examples)
        accuracy = correct / len(test_examples)
        
        print(f"Trial {s} results: Acc: {accuracy:.4f}, Avg Tokens: {avg_num_tokens:.2f}, Avg Branches: {avg_branch_events:.2f}, Time: {t:.2f}s")
        
        all_accuracy.append(accuracy)
        all_branch_events.append(avg_branch_events)
        all_time.append(t)
        all_token_num.append(avg_num_tokens)

    # --- 7. Final Aggregation and Saving (Dynamic filename) ---
    final_accuracy = sum(all_accuracy) / args.num_trials
    final_branch_events = sum(all_branch_events) / args.num_trials
    final_time = sum(all_time) / args.num_trials
    final_token_num = sum(all_token_num) / args.num_trials

    answers.insert(0, {
        'accuracy' : final_accuracy, 
        "average branch events" : final_branch_events, 
        'average new tokens': final_token_num, 
        'time': final_time
    })

    # Dynamic filename based on args
    base_fname = f"{args.dataset_name}_{args.mode}_{args.model_name}"
    
    if args.mode == 'dts':
        dts_params = (
            f"_entro{fmt_float(args.entropy_threshold)}"
            f"_k{args.branch_top_k}"
            f"_max_active_hyps{args.max_active_hyps}"
        )
        base_fname += dts_params

    common_params_list = [
        f"_temp{fmt_float(args.temperature)}",
        f"_trials{args.num_trials}",
        f"_seed{args.seed}"
    ]
    if args.tag:
        common_params_list.append(f"_{args.tag}")
    common_params_list.append(".json")

    common_params = "".join(common_params_list)
    fname = base_fname + common_params

    # Use output dir from config
    out_dir = Path(output_config["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"\nSaved final results to: {out_path}")
    print(f"Final Avg Accuracy: {final_accuracy:.4f}")

    if args.mode == "dts":
        cand_fname = fname.replace(".json", "_candidates_aime25_1.5b.json")
        cand_out_path = out_dir / cand_fname
        with open(cand_out_path, "w", encoding="utf-8") as f:
            json.dump(all_detailed_candidates_only, f, ensure_ascii=False, indent=2)
        print(f"Saved candidates-only results to: {cand_out_path}")

if __name__ == "__main__":
    main()
