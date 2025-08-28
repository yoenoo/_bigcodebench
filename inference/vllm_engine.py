import uuid
from tqdm.asyncio import tqdm
import asyncio
from typing import Any, Dict, List, Tuple, Union, AsyncIterator
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def init_engine(model_path: str, dtype: str, **kwargs: Any) -> AsyncLLMEngine:
  engine_args = AsyncEngineArgs(model=model_path, dtype=dtype, **kwargs)
  return AsyncLLMEngine.from_engine_args(engine_args)


async def _generate_one(
  engine: AsyncLLMEngine,
  tokenizer,
  prompt: str,
  n_samples: int = 1,
  **sampling_kwargs: Any,
) -> List[str]:
  sp = SamplingParams(
    n=n_samples,
    **sampling_kwargs,
  )

  req_id = uuid.uuid4()
  generator = engine.generate(prompt, sp, req_id)

  outputs = []
  async for output in generator:
    for o in output.outputs:
      if not o.finished():
        continue
      outputs.append(o.text)

  return outputs


async def run_batch_inference(
  engine: AsyncLLMEngine,
  tokenizer,
  dataset, 
  n_samples: int,
  parse_fn,
  **sampling_kwargs: Any,
) -> None:
  async def worker(example: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    completions = await _generate_one(
      engine=engine,
      tokenizer=tokenizer,
      prompt=example["prompt"],
      n_samples=n_samples,
      **sampling_kwargs,
  )
    return example, completions

  tasks = [asyncio.create_task(worker(ex)) for ex in dataset]
  for fut in tqdm.as_completed(tasks, total=len(tasks)):
    example, completions = await fut
    if parse_fn is not None:
      parse_fn(example, completions)