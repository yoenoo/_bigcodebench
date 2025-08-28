import asyncio
from datasets import load_dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = "Please provide a self-contained Python script that solves the following problem in a markdown code block"

model = "Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(model)

def apply_chat_template(task_prompt):
  task_prompt = tokenizer.apply_chat_template(
    [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": task_prompt},
    ],
    tokenize=False, add_generation_prompt=True
  )
  return task_prompt

dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")
dataset = dataset.map(lambda x: {"prompt": apply_chat_template(x["complete_prompt"])})

from vllm_engine import run_batch_inference, init_engine

target_path = "bcb_results/Qwen--Qwen3-14B--main--bigcodebench-complete--vllm-1.0-1-sanitized_calibrated_eval_results.jsonl"

engine = init_engine(model, dtype="bfloat16")

samples = asyncio.run(run_batch_inference(
  engine,
  tokenizer,
  dataset,
  n_samples=2,
  max_tokens=16384,
  target_path=target_path,
  parse_fn=lambda e, cs: print(cs[0]),
))
# print(len(samples))
# print(samples[0])
# exit()

## TODO: evals
from bigcodebench.evaluate import evaluate
evaluate(
  split="complete",
  subset="full",
  samples = target_path,
  no_execute = False,
  execution = "local",
  selective_evaluate = "",
  pass_k = "1,5,10",
  save_pass_rate = True,
  # calibrated: bool = True,
  parallel = -1,
  # min_time_limit: float = 1,
  # max_as_limit: int = 30*1024,
  # max_data_limit: int = 30*1024,
  # max_stack_limit: int = 10,
  # check_gt_only: bool = False,
  # no_gt: bool = False,
)