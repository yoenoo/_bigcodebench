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

engine = init_engine(model, dtype="bfloat16")

outputs = asyncio.run(run_batch_inference(
  engine,
  tokenizer,
  dataset, 
  n_samples=2,
  max_tokens=16384,
  parse_fn=lambda e, cs: print(cs[0]),
))
print(outputs)

exit()






from bigcodebench.provider import make_model
from bigcodebench.data import get_bigcodebench 
from bigcodebench.sanitize import sanitize

model = "Qwen/Qwen3-14B"
backend = "vllm"
subset = "full"
split = "complete"
temperature = 1.0
max_new_tokens = 20000
max_model_len = max_new_tokens + 1024 + 512
instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
skip_prefill = False
revision = "main"
trust_remote_code = False
tokenizer_name = None
tokenizer_legacy = False
tp = 1
greedy = False

dataset = get_bigcodebench(subset=subset)
batch_prompts = []
batch_task_ids = []
batch_entry_points = []
for _, (task_id, task) in enumerate(dataset.items()):
  prompt = task["complete_prompt"]
  batch_prompts.append(prompt)
  batch_task_ids.append(task_id)
  batch_entry_points.append(task["entry_point"])

  if len(batch_prompts) >= 8:
    break

model = make_model(
  model=model,
  backend=backend,
  subset=subset,
  split=split,
  # lora_path=lora_path,
  temperature=temperature,
  max_new_tokens=max_new_tokens,
  max_model_len=max_model_len,
  # reasoning_effort=reasoning_effort,
  # reasoning_budget=reasoning_budget,
  # reasoning_beta=reasoning_beta,
  instruction_prefix=instruction_prefix,
  response_prefix=response_prefix,
  prefill=False,
  # prefill=not skip_prefill,
  # base_url=base_url,
  tp=tp,
  revision=revision,
  trust_remote_code=trust_remote_code,
  # direct_completion=direct_completion,
  tokenizer_name=tokenizer_name,
  tokenizer_legacy=tokenizer_legacy
)

outputs = model.codegen(
  batch_prompts,
  do_sample=not greedy,
  num_samples=1,
)

from pprint import pprint

for output in outputs:
  print(output)
  print("#"*100)
  break

# print(instruction_prefix)
# for output in outputs:
#   print(batch_prompts[0])
#   print("#"*100)
#   print(output)
#   print("#"*100)
#   break


samples = []
for task_id, content, entry_point, task_outputs in zip(batch_task_ids, batch_prompts, batch_entry_points, outputs):
  samples.extend([
    dict(task_id=task_id, solution=sanitize(content+completion, entry_point), raw_solution=content+completion)
    for completion in task_outputs
  ])

pprint(samples[0])
exit()

samples = run_codegen(
  split=split,
  subset=subset,
  **model_kwargs,
)
