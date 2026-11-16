from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import MethodConfig
from .runtime import load_local_config

ALLOWED_VALUES: dict[str, set[Any]] = {
    "name": {"huber", "cauchy", "adaptive", "weighted_nls"},
    "loss_scale_m": {4.0, 6.0, 8.0, 10.0, 12.0},
    "reliability_threshold": {0.10, 0.20, 0.30, 0.40},
    "use_temporal_weight": {True, False},
    "max_irls_iterations": {6, 10, 14},
}


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal["name", "loss_scale_m", "reliability_threshold",
                   "use_temporal_weight", "max_irls_iterations"]
    value: Any

    @field_validator("value")
    @classmethod
    def ensure_allowlisted(cls, value: Any, info):
        field = info.data.get("field")
        if field == "name":
            valid_type = isinstance(value, str)
        elif field == "use_temporal_weight":
            valid_type = type(value) is bool
        elif field == "max_irls_iterations":
            valid_type = type(value) is int
        else:
            valid_type = type(value) in (int, float)
        if not valid_type or field not in ALLOWED_VALUES:
            raise ValueError("incorrect proposal value type")
        if value not in ALLOWED_VALUES[field]:
            raise ValueError(f"value is not allowlisted for {field}")
        return value


class TrialProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis: str = Field(min_length=8, max_length=300)
    change: Change
    rationale: str = Field(min_length=8, max_length=400)


def validate_proposal(raw: dict[str, Any]) -> TrialProposal:
    return TrialProposal.model_validate(raw)


def apply_proposal(current: MethodConfig, proposal: TrialProposal) -> MethodConfig:
    field, value = proposal.change.field, proposal.change.value
    if getattr(current, field) == value:
        raise ValueError("proposal is a no-op")
    if field == "loss_scale_m" and current.name in {"nls", "weighted_nls"} and not current.use_temporal_weight:
        raise ValueError("scale is inactive for squared loss without temporal weighting")
    if field in {"reliability_threshold", "max_irls_iterations"} and current.name != "adaptive":
        raise ValueError("control is reserved for adaptive reweighting")
    return MethodConfig(**{**asdict(current), field: value})


def scripted_proposal(iteration_index: int, current: MethodConfig, summary: dict) -> TrialProposal:
    """Legacy fixed order, retained as an explicitly non-adaptive comparison."""
    schedule = [("name", "huber"), ("name", "cauchy"), ("name", "adaptive"),
                ("use_temporal_weight", True)]
    field, value = schedule[min(iteration_index - 1, 3)]
    return validate_proposal({"hypothesis": "Fixed scheduled intervention may improve the localization error tail.",
                              "change": {"field": field, "value": value},
                              "rationale": "Fixed schedule baseline; no result-driven proposal selection is claimed."})


def diagnostic_proposal(current: MethodConfig, summary: dict,
                        history: list[dict] | None = None) -> TrialProposal:
    """Transparent results-driven local strategy, never presented as LLM inference."""
    worst = summary["worst_slices"][0]
    if current.name in {"nls", "weighted_nls"}:
        options = [("name", "huber"), ("name", "cauchy")]
    elif float(worst.get("mobile_rate", 0)) > 0 and not current.use_temporal_weight:
        options = [("use_temporal_weight", True), ("name", "adaptive")]
    elif summary["metrics"]["cep90_m"] > 2 * max(summary["metrics"]["median_error_m"], .01):
        options = [("name", "cauchy"), ("loss_scale_m", 4.0), ("name", "adaptive")]
    else:
        options = [("name", "adaptive"), ("loss_scale_m", 6.0), ("loss_scale_m", 10.0)]
    options += [("name", n) for n in ["huber", "cauchy", "adaptive", "weighted_nls"]]
    attempted = [x.get("candidate_method") for x in history or []]
    for field, value in options:
        proposal = validate_proposal({
            "hypothesis": f"An evidence-conditioned {field} adjustment may reduce the worst-slice error tail.",
            "change": {"field": field, "value": value},
            "rationale": f"Worst validation slice {worst['slice']}: CEP90={worst['cep90_m']:.3f}m; avoid previously attempted configurations."[:400],
        })
        try:
            candidate = apply_proposal(current, proposal)
        except ValueError:
            continue
        if candidate.to_dict() not in attempted:
            return proposal
    raise RuntimeError("diagnostic policy has no untried effective change")


def request_model_proposal(summary: dict, current: MethodConfig, iteration_index: int,
                           history: list[dict] | None = None,
                           feedback: str | None = None) -> tuple[dict, dict]:
    """One bounded inference. No shell, file access or hidden test data is sent."""
    from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, proposal_prompt
    load_local_config()
    provider = os.getenv("AGENT_PROVIDER", "openai")
    model = os.getenv("AGENT_MODEL")
    if not model:
        raise RuntimeError("Set AGENT_MODEL in the explicit project configuration")
    prompt = proposal_prompt(summary, current.to_dict(), iteration_index, history)
    if feedback:
        prompt += "\nPrevious proposal was rejected by the validator: " + feedback + "\nReturn a different legal proposal."
    started = time.perf_counter()
    timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "90"))
    if provider == "openai":
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
                        timeout=timeout, max_retries=0)
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=1200,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
    elif provider == "anthropic":
        import anthropic
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("Anthropic credentials not configured")
        client = anthropic.Anthropic(timeout=timeout, max_retries=0)
        response = client.messages.create(model=model, max_tokens=1200, temperature=0,
                                         system=SYSTEM_PROMPT,
                                         messages=[{"role": "user", "content": prompt}])
        text = "".join(block.text for block in response.content if block.type == "text")
        finish_reason = response.stop_reason
    else:
        raise ValueError("AGENT_PROVIDER must be openai or anthropic")
    trace = {"provider": provider, "model": model, "response_id": response.id,
             "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT,
             "user_prompt": prompt, "raw_response": text,
             "finish_reason": finish_reason, "usage": response.usage.model_dump() if response.usage else {},
             "inference_seconds": time.perf_counter() - started}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = {"invalid_json_response": True}
    return raw, trace


def llm_proposal(summary: dict, current: MethodConfig, iteration_index: int) -> TrialProposal:
    raw, _ = request_model_proposal(summary, current, iteration_index)
    return validate_proposal(raw)
