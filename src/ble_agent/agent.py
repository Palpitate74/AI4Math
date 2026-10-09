from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .models import MethodConfig

ALLOWED_VALUES: dict[str, set[Any]] = {
    "name": {"huber", "cauchy", "adaptive"},
    "loss_scale_m": {4.0, 6.0, 8.0, 10.0, 12.0},
    "reliability_threshold": {0.10, 0.20, 0.30, 0.40},
    "use_temporal_weight": {True},
    "max_irls_iterations": {6, 10, 14},
}


class Change(BaseModel):
    field: Literal[
        "name",
        "loss_scale_m",
        "reliability_threshold",
        "use_temporal_weight",
        "max_irls_iterations",
    ]
    value: Any

    @field_validator("value")
    @classmethod
    def ensure_allowlisted(cls, value: Any, info):
        field_name = info.data.get("field")
        if field_name and value not in ALLOWED_VALUES[field_name]:
            raise ValueError(f"{value!r} is not allowlisted for {field_name}")
        return value


class TrialProposal(BaseModel):
    hypothesis: str = Field(min_length=8, max_length=300)
    change: Change
    rationale: str = Field(min_length=8, max_length=400)


def validate_proposal(raw: dict[str, Any]) -> TrialProposal:
    """Validate that a proposal changes exactly one safe experimental control."""
    return TrialProposal.model_validate(raw)


def apply_proposal(current: MethodConfig, proposal: TrialProposal) -> MethodConfig:
    updates = {proposal.change.field: proposal.change.value}
    if proposal.change.field == "name" and proposal.change.value in {"huber", "cauchy", "adaptive"}:
        return MethodConfig(
            name=proposal.change.value,
            loss_scale_m=current.loss_scale_m,
            reliability_threshold=current.reliability_threshold,
            max_irls_iterations=current.max_irls_iterations,
            use_temporal_weight=current.use_temporal_weight,
        )
    return MethodConfig(**{**asdict(current), **updates})


def scripted_proposal(iteration_index: int, current: MethodConfig, summary: dict) -> TrialProposal:
    """Deterministic fallback policy for fully reproducible no-key execution."""
    del summary
    schedule = [
        {
            "hypothesis": "Heavy-tailed NLOS ranges should be suppressed by a robust loss.",
            "change": {"field": "name", "value": "huber"},
            "rationale": "The baseline's high CEP90 tail indicates residual outliers.",
        },
        {
            "hypothesis": "A Cauchy loss may downweight the largest residuals more strongly.",
            "change": {"field": "name", "value": "cauchy"},
            "rationale": "The remaining worst slice has elevated outlier contamination.",
        },
        {
            "hypothesis": "Alternating residual-based reliability updates should limit persistently inconsistent helper reports.",
            "change": {"field": "name", "value": "adaptive"},
            "rationale": "Robust loss alone cannot combine helper metadata and residual history.",
        },
        {
            "hypothesis": "Temporal stability weighting should reduce error from drifting helper-device reports.",
            "change": {"field": "use_temporal_weight", "value": True},
            "rationale": "The worst validation slice includes a high mobile-helper rate.",
        },
    ]
    raw = schedule[min(iteration_index - 1, len(schedule) - 1)]
    return validate_proposal(raw)


def llm_proposal(summary: dict, current: MethodConfig, iteration_index: int) -> TrialProposal:
    """Ask Claude for a JSON proposal. Credentials are resolved by the official SDK."""
    import anthropic

    from .prompts import SYSTEM_PROMPT, proposal_prompt

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "No explicit Anthropic credential found. Use --policy scripted or authenticate before using --policy api."
        )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": proposal_prompt(summary, current.to_dict(), iteration_index),
            }
        ],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model refused to make a safe bounded experiment proposal.")
    text = "".join(block.text for block in response.content if block.type == "text")
    return validate_proposal(json.loads(text))
