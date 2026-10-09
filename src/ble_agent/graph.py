from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import apply_proposal, llm_proposal, scripted_proposal
from .experiment import accepted_against, run_experiment, validation_summary
from .logging import persist_iteration
from .models import MethodConfig


class AgentState(TypedDict, total=False):
    policy: str
    iteration_index: int
    current_method: dict
    current_experiment_id: str
    summary: dict
    proposal: dict
    candidate_method: dict
    candidate_experiment_id: str
    decision: dict
    record: dict


def inspect_results(state: AgentState) -> AgentState:
    return {"summary": validation_summary(state["current_experiment_id"])}


def propose_trial(state: AgentState) -> AgentState:
    current = MethodConfig(**state["current_method"])
    if state["policy"] == "api":
        proposal = llm_proposal(state["summary"], current, state["iteration_index"])
    else:
        proposal = scripted_proposal(state["iteration_index"], current, state["summary"])
    return {"proposal": proposal.model_dump()}


def validate_proposal(state: AgentState) -> AgentState:
    from .agent import TrialProposal

    proposal = TrialProposal.model_validate(state["proposal"])
    candidate = apply_proposal(MethodConfig(**state["current_method"]), proposal)
    return {"candidate_method": candidate.to_dict()}


def run_allowed_experiment(state: AgentState) -> AgentState:
    method = MethodConfig(**state["candidate_method"])
    result = run_experiment(method, "validation", label=f"validation-agent-{state['iteration_index']}-{method.name}")
    return {"candidate_experiment_id": result.experiment_id}


def evaluate_decision(state: AgentState) -> AgentState:
    current = validation_summary(state["current_experiment_id"])
    candidate = validation_summary(state["candidate_experiment_id"])
    accepted, reason = accepted_against(current, candidate)
    return {
        "decision": {
            "accepted": accepted,
            "reason": reason,
            "current_metrics": current["metrics"],
            "candidate_metrics": candidate["metrics"],
        }
    }


def persist_iteration_node(state: AgentState) -> AgentState:
    record = persist_iteration(
        {
            "iteration_index": state["iteration_index"],
            "policy": state["policy"],
            "input_experiment_id": state["current_experiment_id"],
            "candidate_experiment_id": state["candidate_experiment_id"],
            "diagnosis": state["summary"],
            "proposal": state["proposal"],
            "candidate_method": state["candidate_method"],
            "decision": state["decision"],
        }
    )
    next_state: AgentState = {"record": record}
    if state["decision"]["accepted"]:
        next_state["current_method"] = state["candidate_method"]
        next_state["current_experiment_id"] = state["candidate_experiment_id"]
    return next_state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("inspect_results", inspect_results)
    graph.add_node("propose_trial", propose_trial)
    graph.add_node("validate_proposal", validate_proposal)
    graph.add_node("run_allowed_experiment", run_allowed_experiment)
    graph.add_node("evaluate_decision", evaluate_decision)
    graph.add_node("persist_iteration", persist_iteration_node)
    graph.add_edge(START, "inspect_results")
    graph.add_edge("inspect_results", "propose_trial")
    graph.add_edge("propose_trial", "validate_proposal")
    graph.add_edge("validate_proposal", "run_allowed_experiment")
    graph.add_edge("run_allowed_experiment", "evaluate_decision")
    graph.add_edge("evaluate_decision", "persist_iteration")
    graph.add_edge("persist_iteration", END)
    return graph.compile()


def run_iteration(
    *,
    policy: str,
    iteration_index: int,
    current_method: MethodConfig,
    current_experiment_id: str,
) -> AgentState:
    if policy not in {"scripted", "api"}:
        raise ValueError("policy must be scripted or api")
    graph = build_graph()
    return graph.invoke(
        {
            "policy": policy,
            "iteration_index": iteration_index,
            "current_method": current_method.to_dict(),
            "current_experiment_id": current_experiment_id,
        }
    )
