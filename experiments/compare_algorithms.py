"""
Algorithm Comparison: GRPO vs DPO vs PPO on GSM8K
===================================================
Trains Qwen2.5-0.5B with each algorithm for the same number of steps
and compares GSM8K accuracy, training stability, and compute efficiency.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import copy, random, time, json, argparse, logging, os, re
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s -- %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_ITERATIONS = 300
GROUP_SIZE = 4
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.9
LR = 1e-6
BETA = 0.01
EPSILON = 0.2
LOG_INTERVAL = 25
EVAL_INTERVAL = 50

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Solve the problem step by step. "
    "End your response with '#### <answer>' where <answer> is the numeric result."
)

def format_prompt(question):
    return f"{SYSTEM_PROMPT}\n\nProblem: {question}\n\nSolution:"

def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")
    train = list(dataset["train"])
    test = list(dataset["test"])
    random.shuffle(train)
    return train, test

def extract_answer(text):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if m: return m.group(1).replace(',', '')
    m = re.search(r'answer\s+(?:is|:)\s*(-?[\d,]+\.?\d*)', text, re.I)
    if m: return m.group(1).replace(',', '')
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    return nums[-1].replace(',', '') if nums else None

def normalize(s):
    if s is None: return None
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{v:.4f}".rstrip('0')
    except: return s.strip().lower()

def score(completion, ground_truth):
    p = normalize(extract_answer(completion))
    c = normalize(extract_answer(ground_truth))
    if p is None or c is None: return 0.0
    return 1.0 if p == c else 0.0

def generate_completions(model, tokenizer, prompt, n, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    completions, comp_ids_list = [], []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            ids = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            completions.append(text)
            comp_ids_list.append(ids)
    return completions, comp_ids_list, inputs

def get_log_probs(model, inputs, comp_ids):
    full_ids = torch.cat([inputs["input_ids"][0], comp_ids]).unsqueeze(0)
    logits = model(full_ids).logits[0, :-1]
    lp_all = F.log_softmax(logits, dim=-1)
    targets = full_ids[0, 1:]
    lp = lp_all.gather(1, targets.unsqueeze(1)).squeeze(1)
    start = inputs["input_ids"].shape[1] - 1
    return lp[start:]

def grpo_step(model, ref_model, tokenizer, optimizer, prompt, ground_truth, device):
    completions, comp_ids_list, inputs = generate_completions(model, tokenizer, prompt, GROUP_SIZE, device)
    rewards = torch.tensor([score(c, ground_truth) for c in completions])
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    for comp_ids, adv in zip(comp_ids_list, advantages):
        lp = get_log_probs(model, inputs, comp_ids)
        ref_lp = get_log_probs(ref_model, inputs, comp_ids)
        kl = (torch.exp(ref_lp - lp) - (ref_lp - lp) - 1).mean()
        loss = -(lp * adv.to(device)).mean() + BETA * kl
        total_loss = total_loss + loss
    total_loss = total_loss / GROUP_SIZE
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return rewards.mean().item(), total_loss.item()

def ppo_step(model, ref_model, tokenizer, optimizer, prompt, ground_truth, device):
    completions, comp_ids_list, inputs = generate_completions(model, tokenizer, prompt, GROUP_SIZE, device)
    rewards = torch.tensor([score(c, ground_truth) for c in completions])
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    old_lps = []
    with torch.no_grad():
        for comp_ids in comp_ids_list:
            old_lps.append(get_log_probs(model, inputs, comp_ids).detach())
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    for comp_ids, adv, old_lp in zip(comp_ids_list, advantages, old_lps):
        lp = get_log_probs(model, inputs, comp_ids)
        ref_lp = get_log_probs(ref_model, inputs, comp_ids)
        ratio = torch.exp(lp - old_lp)
        obj1 = ratio * adv.to(device)
        obj2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * adv.to(device)
        pg_loss = -torch.min(obj1, obj2).mean()
        kl = (torch.exp(ref_lp - lp) - (ref_lp - lp) - 1).mean()
        loss = pg_loss + BETA * kl
        total_loss = total_loss + loss
    total_loss = total_loss / GROUP_SIZE
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return rewards.mean().item(), total_loss.item()

def dpo_step(model, ref_model, tokenizer, optimizer, prompt, ground_truth, device):
    completions, comp_ids_list, inputs = generate_completions(model, tokenizer, prompt, 2, device)
    r0 = score(completions[0], ground_truth)
    r1 = score(completions[1], ground_truth)
    if r0 == r1:
        return (r0 + r1) / 2, 0.0
    if r0 > r1:
        chosen_ids, rejected_ids = comp_ids_list[0], comp_ids_list[1]
    else:
        chosen_ids, rejected_ids = comp_ids_list[1], comp_ids_list[0]
    lp_chosen = get_log_probs(model, inputs, chosen_ids)
    lp_rejected = get_log_probs(model, inputs, rejected_ids)
    ref_chosen = get_log_probs(ref_model, inputs, chosen_ids)
    ref_rejected = get_log_probs(ref_model, inputs, rejected_ids)
    logits = BETA * ((lp_chosen.sum() - ref_chosen.sum()) - (lp_rejected.sum() - ref_rejected.sum()))
    loss = -F.logsigmoid(logits)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return (r0 + r1) / 2, loss.item()

def evaluate(model, tokenizer, problems, n=100, device="cuda"):
    model.eval()
    correct = 0
    sample = random.sample(problems, min(n, len(problems)))
    with torch.no_grad():
        for p in sample:
            prompt = format_prompt(p["question"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
            ids = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            correct += score(text, p["answer"])
    model.train()
    return correct / len(sample)

def train(algorithm, train_problems, test_problems, device):
    logger.info(f"\n{'='*50}\nTraining: {algorithm.upper()}\n{'='*50}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    step_fns = {"grpo": grpo_step, "ppo": ppo_step, "dpo": dpo_step}
    step_fn = step_fns[algorithm]
    history = {"rewards": [], "losses": [], "eval_accs": [], "eval_steps": []}
    t_start = time.time()
    for i in range(NUM_ITERATIONS):
        problem = train_problems[i % len(train_problems)]
        prompt = format_prompt(problem["question"])
        reward, loss = step_fn(model, ref_model, tokenizer, optimizer, prompt, problem["answer"], device)
        history["rewards"].append(reward)
        history["losses"].append(loss)
        if (i + 1) % LOG_INTERVAL == 0:
            w_acc = np.mean(history["rewards"][-LOG_INTERVAL:])
            logger.info(f"[{algorithm.upper()}] iter={i+1} window_acc={w_acc:.3f} loss={loss:.4f}")
        if (i + 1) % EVAL_INTERVAL == 0:
            acc = evaluate(model, tokenizer, test_problems, n=100, device=device)
            history["eval_accs"].append(acc)
            history["eval_steps"].append(i + 1)
            logger.info(f"[{algorithm.upper()}] EVAL iter={i+1} acc={acc:.3f}")
    elapsed = time.time() - t_start
    final_acc = evaluate(model, tokenizer, test_problems, n=200, device=device)
    logger.info(f"[{algorithm.upper()}] Final accuracy: {final_acc:.3f} ({elapsed:.0f}s)")
    return {"algorithm": algorithm, "final_acc": final_acc, "elapsed_s": elapsed, "history": history}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=["grpo", "dpo", "ppo"])
    parser.add_argument("--output", default="experiments/results.json")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    train_problems, test_problems = load_gsm8k()
    results = []
    for algo in args.algorithms:
        result = train(algo, train_problems, test_problems, device)
        results.append(result)
        os.makedirs("experiments", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*60)
    print(f"{'Algorithm':<10} {'Final Acc':>10} {'Time (min)':>12}")
    print("-"*60)
    for r in results:
        print(f"{r['algorithm'].upper():<10} {r['final_acc']:>10.3f} {r['elapsed_s']/60:>12.1f}")
    print("="*60)
