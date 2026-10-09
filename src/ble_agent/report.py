from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .logging import read_iterations

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
REPORT_DIR = ARTIFACTS / "report"
FIGURE_DIR = ARTIFACTS / "figures"


def _read_experiment(experiment_id: str) -> dict:
    return json.loads((ARTIFACTS / "experiments" / f"{experiment_id}.json").read_text(encoding="utf-8"))


def _commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "uncommitted"


def _plot_error_tail(experiments: list[dict]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = []
    cep90 = []
    medians = []
    for record in experiments:
        labels.append(record["method"]["name"])
        cep90.append(record["metrics"]["cep90_m"])
        medians.append(record["metrics"]["median_error_m"])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, cep90, width=0.36, label="CEP90")
    ax.bar(x + 0.18, medians, width=0.36, label="Median error")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Position error (m)")
    ax.set_title("Held-out test localization error")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / "test_error_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_iteration_curve(records: list[dict]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    points = []
    labels = []
    for record in records:
        candidate = record["decision"]["candidate_metrics"]["cep90_m"]
        current = record["decision"]["current_metrics"]["cep90_m"]
        points.extend([current, candidate])
        labels.extend([f"{record['iteration_index']}: before", f"{record['iteration_index']}: trial"])
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(range(len(points)), points, marker="o", color="#0067a5")
    ax.set_xticks(range(len(points)), labels, rotation=25, ha="right")
    ax.set_ylabel("Validation CEP90 (m)")
    ax.set_title("LangGraph agent iteration trace")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURE_DIR / "iteration_cep90.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _paragraph(text: str, styles, style: str = "BodyText"):
    return Paragraph(text, styles[style])


def build_report() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = read_iterations()
    if not records:
        raise RuntimeError("No agent iteration log exists. Run the baseline and agent scripts first.")

    state = json.loads((ARTIFACTS / "agent_state.json").read_text(encoding="utf-8"))
    baseline_test = _read_experiment("test-baseline-nls")
    final_test = _read_experiment(f"test-final-{state['current_method']['name']}")
    iteration_figure = _plot_iteration_curve(records)
    test_figure = _plot_error_tail([baseline_test, final_test])

    output = REPORT_DIR / "robust_ble_localization_agent_report.pdf"
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm
    )
    styles = getSampleStyleSheet()
    styles["Title"].fontName = "Helvetica-Bold"
    story = []
    story += [
        _paragraph("Robust BLE Crowdsourced Localization Experiment Agent", styles, "Title"),
        Spacer(1, 0.3 * cm),
        _paragraph("Course assignment progress report and reproducible experimental result", styles, "Heading2"),
        _paragraph(
            "<b>Scope.</b> This study addresses Huawei Challenge 1 as a bounded applied-mathematics subproblem: estimate a 2-D offline-device position from noisy crowdsourced BLE reports. It does not claim to meet Huawei production KPIs or evaluate proprietary data.",
            styles,
        ),
        Spacer(1, 0.2 * cm),
        _paragraph(
            "<b>Headline result.</b> A LangGraph-controlled, validation-only iterative search selected "
            f"<b>{final_test['method']['name']}</b>. On the held-out synthetic test split, CEP90 changed from "
            f"<b>{baseline_test['metrics']['cep90_m']:.2f} m</b> (ordinary nonlinear least squares) to "
            f"<b>{final_test['metrics']['cep90_m']:.2f} m</b>. The conclusion is conditional on the stated simulator assumptions.",
            styles,
        ),
        Spacer(1, 0.4 * cm),
        _paragraph("1. Mathematical formulation", styles, "Heading1"),
        _paragraph(
            "For helper reports with positions l_i and RSSI-derived ranges d_i=f(r_i), the baseline solves min_x Σ_i (d_i − ||x−l_i||_2)^2. The robust alternatives replace the squared loss with Huber/Cauchy losses. The adaptive method alternates robust coordinate estimation with reliability updates derived from metadata quality, residual consistency, and temporal range stability.",
            styles,
        ),
        _paragraph("2. Agent architecture and guardrails", styles, "Heading1"),
        _paragraph(
            "The LangGraph state graph is inspect_results → propose_trial → validate_proposal → run_allowed_experiment → evaluate_decision → persist_iteration. The LLM/policy can propose only one allowlisted change; deterministic Python tools simulate and score experiments. The agent never receives held-out test metrics during tuning, has no shell tool, and cannot execute arbitrary code.",
            styles,
        ),
        _paragraph(
            f"The recorded run used <b>{records[0]['policy']}</b> policy. Scripted mode is a reproducible fallback; an API-backed run should only be described as LLM-driven when API credentials are available and an actual API trace is retained.",
            styles,
        ),
        _paragraph("3. Experimental protocol", styles, "Heading1"),
        _paragraph(
            "Each synthetic scenario samples true device positions, 10/15/20 helper devices, RSSI-range noise, NLOS positive-bias outliers, stale/mobile helper coordinates, and layout degradation. Random seeds and scenario configurations are stored with each artifact. The train/validation/test scenario files use distinct seed ranges. Validation CEP90 is the primary decision metric; median-error degradation is constrained before a candidate is accepted.",
            styles,
        ),
        PageBreak(),
        _paragraph("4. Agent iterations", styles, "Heading1"),
        Image(str(iteration_figure), width=16.5 * cm, height=8.2 * cm),
    ]

    rows = [["Run", "Proposal", "Accepted?", "CEP90 before", "CEP90 trial", "Decision"]]
    for record in records:
        proposal = record["proposal"]
        decision = record["decision"]
        rows.append(
            [
                str(record["iteration_index"]),
                f"{proposal['change']['field']}={proposal['change']['value']}",
                "Yes" if decision["accepted"] else "No",
                f"{decision['current_metrics']['cep90_m']:.2f}",
                f"{decision['candidate_metrics']['cep90_m']:.2f}",
                decision["reason"],
            ]
        )
    table = Table(rows, colWidths=[0.8 * cm, 3.0 * cm, 1.3 * cm, 2.0 * cm, 1.8 * cm, 8.0 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#005b96")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eaf3f8")]),
            ]
        )
    )
    story += [table, Spacer(1, 0.4 * cm), _paragraph("5. Frozen test evaluation", styles, "Heading1"), Image(str(test_figure), width=16.5 * cm, height=9.2 * cm)]

    metrics_rows = [["Method", "Median m", "CEP90 m", "Failure >30m", "Mobile F1", "Mean runtime ms"]]
    for record in [baseline_test, final_test]:
        metrics = record["metrics"]
        metrics_rows.append(
            [
                record["method"]["name"],
                f"{metrics['median_error_m']:.2f}",
                f"{metrics['cep90_m']:.2f}",
                f"{metrics['failure_rate_over_30m']:.1%}",
                f"{metrics['mobile_detection']['f1']:.2f}",
                f"{metrics['mean_runtime_ms']:.2f}",
            ]
        )
    metric_table = Table(metrics_rows, colWidths=[3.0 * cm, 2.2 * cm, 2.2 * cm, 3.0 * cm, 2.2 * cm, 3.1 * cm], repeatRows=1)
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#005b96")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story += [metric_table, Spacer(1, 0.4 * cm)]
    story += [
        _paragraph("6. Limitations and reproducibility", styles, "Heading1"),
        _paragraph(
            "These results are generated from a simplified 2-D simulator. Real BLE deployments additionally require RSSI calibration by device/environment, map and floor constraints, privacy-safe telemetry, delayed reports, and real movement traces. The Huawei target of CEP90 ≤30 m and 20% mobile-detection improvement must be evaluated using the official validation data and protocol; it is not claimed here.",
            styles,
        ),
        _paragraph(
            f"Reproduce with: <font name='Courier'>.venv/bin/python scripts/run_baseline.py; .venv/bin/python scripts/run_agent.py --policy scripted --reset; .venv/bin/python scripts/run_final_test.py; .venv/bin/python scripts/build_report.py</font>. Git revision at report generation: <b>{_commit_hash()}</b>.",
            styles,
        ),
    ]
    doc.build(story)
    return output
