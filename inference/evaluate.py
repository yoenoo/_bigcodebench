from typing import Any, Dict, Optional
from bigcodebench.eval import untrusted_check, PASS
from collections import defaultdict
from tqdm import tqdm

def evaluate_single_sample(
    sample: Dict[str, Any],
    problems: Dict[str, Dict[str, Any]],
    expected_time: Dict[str, Optional[float]],
    calibrated: bool = True,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    min_time_limit: float = 1.0,
    default_gt_time_limit: float = 20.0,
    include_solution: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a single BigCodeBench sample locally.

    Args:
        sample: One record from load_solutions(...). Must contain "task_id" and either
                "solution" or "completion" (used with problem['complete_prompt']).
        problems: Dict task_id -> problem metadata from get_bigcodebench(...).
        expected_time: Dict task_id -> float|None from get_groundtruth(...).
        calibrated: If True, prepend code_prompt + '    pass' (your batch logic).
        *_limits/time: Resource and timeout knobs (same semantics as batch).
        include_solution: If True, include the exact solution string in the return.

    Returns:
        A dict shaped like entries stored in results["eval"][task_id]:
        {
          "task_id": str,
          "status": str,     # PASS / FAIL / etc.
          "details": List[bool] or Any,  # per-test booleans from untrusted_check
          "solution": str (optional)
        }

    Raises:
        KeyError if task_id not found in `problems`.
        ValueError if sample lacks both 'solution' and 'completion'.
    """
    task_id = sample["task_id"]
    problem = problems[task_id]

    # Build solution text (same as your batch path)
    if "solution" in sample:
        solution = sample["solution"]
    elif "completion" in sample:
        solution = problem["complete_prompt"] + sample["completion"]
    else:
        raise ValueError("Sample must contain 'solution' or 'completion'.")

    if calibrated:
        solution = problem["code_prompt"] + "\n    pass\n" + solution

    # Per-task timeout: GT if available else default
    gt_time_limit = expected_time.get(task_id) or default_gt_time_limit

    # Run sandboxed check
    status, details = untrusted_check(
        code=solution,
        test_code=problem["test"],
        entry_point=problem["entry_point"],
        max_as_limit=max_as_limit,
        max_data_limit=max_data_limit,
        max_stack_limit=max_stack_limit,
        min_time_limit=min_time_limit,
        gt_time_limit=gt_time_limit,
    )

    record = {
        "task_id": task_id,
        "status": status,
        "details": details,
    }
    if include_solution:
        record["solution"] = solution
    return record


if __name__ == "__main__":
  from bigcodebench.data import get_bigcodebench, load_solutions

  target_path = "bcb_results/Qwen--Qwen3-14B--main--bigcodebench-complete--vllm-1.0-1-sanitized_calibrated_eval_results.jsonl"
  problems = get_bigcodebench(subset="full")

  results = defaultdict(list)
  for sample in tqdm(load_solutions(target_path), total=10*2):
    # print(sample)
    rec = evaluate_single_sample(sample, problems, {}, calibrated=True)
    task_id = rec["task_id"]
    status = rec["status"]
    details = rec["details"]

    if status not in ["pass", "fail"]:
      raise ValueError(f"Unknown status: {status}")

    status_id = 1 if status == "pass" else 0
    results[task_id].append(status_id)

  pass_at_n = defaultdict(list)
  for task_id, status_list in results.items():
    pass_at_n[task_id] = max(status_list) ## pass@n 

  print(pass_at_n)