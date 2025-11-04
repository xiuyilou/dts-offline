import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from decoding_tree_sketching.kvbatch_decoder import KVBatchEGDT

examples = [
    "Let $\\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of $\\mathcal{B}$. The value of $r^2$ can be written as $\\\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.",
]
groundtruths = [
    "721",
]
reasoning_tail = r" Please reason step by step, and put your final answer within \boxed{}."
seed = 0

# Replace with your model checkpoint if needed
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Decoding hyperparameters
DECODE_CONFIG = {
    "entropy_threshold": 2.5,
    "branch_top_k": 3,
    "max_active_hyps": 12,
    "max_new_tokens": 4096,
    "temperature": 0.6,
}

def main():
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True
    )
    streamer = TextStreamer(tokenizer)

    kvegdt = KVBatchEGDT(model, tokenizer, seed=seed)

    ques_idx = 0
    example = examples[ques_idx]
    groundtruth = groundtruths[ques_idx]
    full_prompt = example + reasoning_tail
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    # Regular output
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=DECODE_CONFIG["max_new_tokens"],
        do_sample=True,     
        temperature=DECODE_CONFIG["temperature"],    
        streamer=streamer,
    )

    # DTS output
    dts_out = kvegdt.generate(
        text,
        entropy_threshold=DECODE_CONFIG["entropy_threshold"],
        branch_top_k=DECODE_CONFIG["branch_top_k"],
        max_active_hyps=DECODE_CONFIG["max_active_hyps"],
        max_new_tokens=DECODE_CONFIG["max_new_tokens"],
        temperature=DECODE_CONFIG["temperature"],
    )
    
    print(f"*** MODEL OUTPUT ***\n{dts_out['text']}")
    # Print generation statistics such as steps, branch events, and sequence length
    print(f"\n*** GENERATION STATS ***\n{dts_out['stats']}")

    print(f"Groundtruth = {groundtruth}, Regular decoding output = {}, DTS output = {dts_out['stats']}")

if __name__ == "__main__":
    main()       
