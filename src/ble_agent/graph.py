from __future__ import annotations

from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .agent import (apply_proposal, diagnostic_proposal, request_model_proposal,
                    scripted_proposal, validate_proposal)
from .experiment import accepted_against, run_experiment, validation_summary
from .logging import persist_iteration, persist_model_event, read_iterations
from .models import MethodConfig


class AgentState(TypedDict, total=False):
    policy: str
    iteration_index: int
    iteration_id: str
    current_method: dict
    current_experiment_id: str
    validation_split: str
    summary: dict
    history: list[dict]
    proposal: dict
    candidate_method: dict
    candidate_experiment_id: str
    decision: dict
    model_traces: list[dict]
    proposal_attempts: list[dict]
    error: str
    record: dict


def inspect_results(state: AgentState) -> AgentState:
    history = [{k: r.get(k) for k in ["iteration_index", "proposal", "candidate_method", "decision", "error"]}
               for r in read_iterations()]
    return {"summary": validation_summary(state["current_experiment_id"]), "history": history}


def propose_trial(state: AgentState) -> AgentState:
    current = MethodConfig(**state["current_method"])
    traces, attempts = [], []
    feedback = None
    for attempt in range(1, 4 if state["policy"] == "api" else 2):
        try:
            if state["policy"] == "api":
                raw, trace = request_model_proposal(state["summary"], current,
                    state["iteration_index"], state["history"], feedback)
                trace["attempt"] = attempt
                traces.append(trace)
                persist_model_event(state["iteration_id"], state["iteration_index"], trace)
            elif state["policy"] == "diagnostic":
                raw = diagnostic_proposal(current, state["summary"], state["history"]).model_dump()
            else:
                raw = scripted_proposal(state["iteration_index"], current, state["summary"]).model_dump()
            proposal = validate_proposal(raw)
            candidate = apply_proposal(current, proposal)
            if any(r.get("candidate_method") == candidate.to_dict() for r in state["history"]):
                raise ValueError("candidate full configuration was already attempted")
            attempts.append({"attempt": attempt, "status": "valid", "proposal": raw})
            return {"proposal": proposal.model_dump(), "candidate_method": candidate.to_dict(),
                    "model_traces": traces, "proposal_attempts": attempts}
        except (ValueError, TypeError) as exc:
            # Only local validation messages are fed back, never transport/credential text.
            feedback = ("Invalid schema, ineffective or repeated change. For nls/weighted_nls WITHOUT temporal weighting, scale is inactive; "
                        "reliability_threshold and max_irls_iterations are allowed ONLY for name=adaptive. "
                        "Choose a different untried method, effective robust loss scale, or temporal toggle; "
                        "check the current method and history before proposing.")
            attempts.append({"attempt": attempt, "status": "invalid", "error_type": type(exc).__name__})
        except Exception as exc:
            # Keep request failures visible without leaking request headers/secrets.
            code = getattr(exc, "status_code", None)
            error = f"Model request failed: {type(exc).__name__}; HTTP status={code}. No scripted fallback was run."
            return {"error": error, "model_traces": traces, "proposal_attempts": attempts}
    return {"error": "No valid untried proposal after bounded validation retries.",
            "model_traces": traces, "proposal_attempts": attempts}


def validate_proposal_node(state: AgentState) -> AgentState:
    proposal = validate_proposal(state["proposal"])
    candidate = apply_proposal(MethodConfig(**state["current_method"]), proposal)
    return {"candidate_method": candidate.to_dict()}


def run_allowed_experiment(state: AgentState) -> AgentState:
    if state["validation_split"] not in {"validation", "midterm_validation"}:
        raise ValueError("agent executor is validation-only")
    method = MethodConfig(**state["candidate_method"])
    result = run_experiment(method, state["validation_split"],
        label=f"validation-agent-{state['iteration_index']}-{method.name}-{uuid4().hex[:8]}")
    return {"candidate_experiment_id": result.experiment_id}


def evaluate_decision(state: AgentState) -> AgentState:
    current = state["summary"]
    candidate = validation_summary(state["candidate_experiment_id"])
    accepted, reason = accepted_against(current, candidate)
    return {"decision": {"accepted": accepted, "reason": reason,
                         "current_metrics": current["metrics"],
                         "candidate_metrics": candidate["metrics"],
                         "candidate_slices": candidate["slices"]}}


def persist_iteration_node(state: AgentState) -> AgentState:
    decision = state.get("decision", {"accepted": False, "reason": state.get("error", "No experiment executed")})
    record = persist_iteration({
        "iteration_id": state["iteration_id"],
        "iteration_index": state["iteration_index"], "policy": state["policy"],
        "status": "error" if state.get("error") else "evaluated",
        "input_experiment_id": state["current_experiment_id"],
        "candidate_experiment_id": state.get("candidate_experiment_id"),
        "diagnosis": state.get("summary", {}), "proposal": state.get("proposal"),
        "candidate_method": state.get("candidate_method"), "decision": decision,
        "model_traces": state.get("model_traces", []),
        "proposal_attempts": state.get("proposal_attempts", []), "error": state.get("error"),
    })
    output: AgentState = {"record": record, "decision": decision}
    if decision["accepted"]:
        output.update(current_method=state["candidate_method"],
                      current_experiment_id=state["candidate_experiment_id"])
    return output


def audited_node(name, node):
    def execute(state):
        try:
            return node(state)
        except Exception as exc:
            # Persist safe failure metadata, never arbitrary exception text/headers.
            return {"error": f"{name} failed: {type(exc).__name__}",
                    "decision": {"accepted": False, "reason": f"{name} failed: {type(exc).__name__}"}}
    return execute


def build_graph():
    graph = StateGraph(AgentState)
    for name, node in [("inspect_results", inspect_results), ("propose_trial", propose_trial),
                       ("validate_proposal", validate_proposal_node),
                       ("run_allowed_experiment", run_allowed_experiment),
                       ("evaluate_decision", evaluate_decision),
                       ("persist_iteration", persist_iteration_node)]:
        graph.add_node(name, node if name == "persist_iteration" else audited_node(name, node))
    graph.add_edge(START, "inspect_results")
    graph.add_conditional_edges("inspect_results", lambda s: "error" if s.get("error") else "valid",
                                {"error": "persist_iteration", "valid": "propose_trial"})
    graph.add_conditional_edges("propose_trial", lambda state: "error" if state.get("error") else "valid",
                                {"error": "persist_iteration", "valid": "validate_proposal"})
    graph.add_conditional_edges("validate_proposal", lambda s: "error" if s.get("error") else "valid",
                                {"error": "persist_iteration", "valid": "run_allowed_experiment"})
    graph.add_conditional_edges("run_allowed_experiment", lambda s: "error" if s.get("error") else "valid",
                                {"error": "persist_iteration", "valid": "evaluate_decision"})
    graph.add_edge("evaluate_decision", "persist_iteration")
    graph.add_edge("persist_iteration", END)
    return graph.compile()


def run_iteration(*, policy: str, iteration_index: int, current_method: MethodConfig,
                  current_experiment_id: str, validation_split: str = "validation") -> AgentState:
    if policy not in {"scripted", "api", "diagnostic"}:
        raise ValueError("unsupported policy")
    if validation_split not in {"validation", "midterm_validation"}:
        raise ValueError("agent cannot run held-out splits")
    if any(r["iteration_index"] == iteration_index for r in read_iterations()):
        raise RuntimeError("Iteration index already has an authoritative decision; recover summary first")
    return build_graph().invoke({"policy": policy, "iteration_index": iteration_index,
        "iteration_id": f"iteration-{iteration_index}-{uuid4().hex}",
        "current_method": current_method.to_dict(), "current_experiment_id": current_experiment_id,
        "validation_split": validation_split})
