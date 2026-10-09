SYSTEM_PROMPT = """You are a cautious applied-mathematics experiment planner.
You are tuning a robust BLE multilateration method on a validation set. You must propose exactly one
allowed change. Never claim production performance and never request test-set metrics.
Allowed changes are: choose Huber or Cauchy loss, set a loss scale from [4, 6, 8, 10, 12],
set a reliability threshold from [0.10, 0.20, 0.30, 0.40], enable temporal reliability weighting,
or change maximum IRLS iterations to [6, 10, 14]. Your response must be JSON only.
"""


def proposal_prompt(summary: dict, current_method: dict, iteration_index: int) -> str:
    return f"""Iteration {iteration_index}. The current validation summary is:
{summary}

Current method:
{current_method}

Return this JSON object only:
{{
  "hypothesis": "short falsifiable rationale",
  "change": {{"field": "one allowed field", "value": "one allowed value"}},
  "rationale": "connect the single change to the worst validation slice"
}}
"""
