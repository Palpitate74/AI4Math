"""Chinese midterm PDF from one isolated run; never execute or tune experiments.

The summary is the source of experiment IDs, policy, protocol and freeze status.
Missing optional evidence is shown, not inferred. ``artifact_base`` exists for
isolated report fixtures; normal callers use artifacts/midterm/<run_id>.
"""
from __future__ import annotations

import io
import json
import math
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, Image, KeepInFrame, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from .runtime import ROOT, validate_identifier

_WIDTH = A4[0] - 84
_HEIGHT = A4[1] - 102
_MISSING = "未记录"


def _text(value) -> str:
    if value is None:
        return _MISSING
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _excerpt(value, limit=500) -> str:
    text = _text(value)
    return text if len(text) <= limit else text[:limit] + " （节录；完整内容见源 JSON/JSONL）"


def _number(value, *, percent=False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return _MISSING
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def _mapping(value) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _flatten(value, prefix="") -> list[tuple[str, str]]:
    rows = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            rows.extend(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    else:
        rows.append((prefix, _text(value)))
    return rows


def _read_object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _load_evidence(run: Path, run_id: str):
    summary = _read_object(run / "summary.json")
    if summary.get("run_id") != run_id:
        raise ValueError("summary.run_id does not match --run-id")
    warnings: list[str] = []
    records = []
    log = run / "iterations" / "agent_iterations.jsonl"
    if not log.exists():
        warnings.append("迭代日志不存在；不能证明执行了任何提案轮次。")
    else:
        for line_number, line in enumerate(log.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record is not an object")
                records.append(row)
            except (ValueError, TypeError):
                warnings.append(f"迭代日志第 {line_number} 行无效；该行未纳入报告。")
    # New runs use immutable decisions as authority; JSONL is a mirror.
    by_id = {r["iteration_id"]: r for r in records if r.get("iteration_id")}
    legacy = [r for r in records if not r.get("iteration_id")]
    for path in (run / "iterations" / "decisions").glob("*.json"):
        record = _read_object(path)
        if not record.get("iteration_id"):
            raise ValueError("Immutable decision lacks iteration_id")
        by_id[record["iteration_id"]] = record
    records = sorted(legacy + list(by_id.values()), key=lambda r: r.get("iteration_index", 0))
    experiments = {}
    directory = run / "experiments"
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            try:
                record = _read_object(path)
                identifier = validate_identifier(path.stem)
                if record.get("experiment_id") != identifier:
                    warnings.append(f"实验 {path.name} 的内部 ID 不匹配，未使用。")
                    continue
                experiments[identifier] = record
            except (OSError, ValueError):
                warnings.append(f"实验 {path.name} 无法读取，未使用。")
    else:
        warnings.append("experiments 目录不存在；指标不可用。")
    references = [summary.get("baseline_validation"), summary.get("final_validation")]
    for key in ("comparisons", "holdout"):
        references += list(_mapping(summary.get(key)).values())
    for row in records:
        references += [row.get("input_experiment_id"), row.get("candidate_experiment_id")]
    for identifier in dict.fromkeys(x for x in references if isinstance(x, str)):
        if identifier not in experiments:
            warnings.append(f"引用的实验 {identifier} 缺失或无效，不能报告其结果。")
    completed = summary.get("completed_iterations")
    if isinstance(completed, int) and completed != len(records):
        warnings.append(f"summary 完成轮数 {completed} 与可读日志轮数 {len(records)} 不一致。")
    logged_errors = sum(bool(r.get("error") or r.get("status") == "error") for r in records)
    if isinstance(summary.get("error_iterations"), int) and summary["error_iterations"] != logged_errors:
        warnings.append("summary 错误轮数与日志错误记录数不一致；两者均保留展示。")
    api_evaluated = sum(r.get("policy") == "api" and r.get("status") == "evaluated" for r in records)
    if isinstance(summary.get("api_evaluated_iterations"), int) and summary["api_evaluated_iterations"] != api_evaluated:
        warnings.append("summary API 已评估轮数与日志不一致；API 策略字段本身不证明成功模型推理。")
    # Trace presence alone is not a successful candidate evaluation. Audit the
    # IDs/raw response independently of policy/status accounting.
    traces = [t for r in records for t in r.get("model_traces", []) if isinstance(t, dict)]
    response_ids = [t.get("response_id") for t in traces]
    if any(not identifier or not t.get("raw_response") for identifier, t in zip(response_ids, traces)):
        warnings.append("部分模型 trace 缺少 response_id 或原始响应；不能全部计为可核验模型响应。")
    if len({i for i in response_ids if isinstance(i, str)}) != len(response_ids):
        warnings.append("模型 response_id 缺失或重复；trace 数不等于独立真实响应数。")
    for row in records:
        for role in ("current", "candidate"):
            identifier = row.get("input_experiment_id" if role == "current" else "candidate_experiment_id")
            record = experiments.get(identifier) if isinstance(identifier, str) else None
            metrics = _mapping(_mapping(row.get("decision")).get(f"{role}_metrics"))
            if record and metrics and metrics != record.get("metrics"):
                warnings.append(f"轮次 {row.get('iteration_index')} 的 {role} 决策指标与实验 JSON 不一致。")
        if row.get("status") == "evaluated" and not row.get("candidate_experiment_id"):
            warnings.append(f"轮次 {row.get('iteration_index')} 标称已评估但没有候选实验引用。")
    final_id = summary.get("final_validation")
    final = experiments.get(final_id) if isinstance(final_id, str) else None
    if final and summary.get("final_method") != final.get("method"):
        warnings.append("summary.final_method 与最终验证实验配置不一致；禁止据此归因最终方法收益。")
    tuning_ids = [summary.get("baseline_validation"), summary.get("final_validation")]
    tuning_ids += list(_mapping(summary.get("comparisons")).values())
    for identifier in tuning_ids:
        record = experiments.get(identifier) if isinstance(identifier, str) else None
        if record and record.get("split") not in {"validation", "midterm_validation"}:
            warnings.append(f"验证比较 {identifier} 的实际 split 不是验证集；不能作为合规调参证据。")
        if record and any(s.get("metadata_mode") == "oracle" for s in record.get("scenarios", []) if isinstance(s, dict)):
            warnings.append(f"验证实验 {identifier} 含 oracle 元数据，不符合 Agent 调参边界。")
    tuning_ids += [r.get("candidate_experiment_id") for r in records]
    validation_records = [experiments[i] for i in dict.fromkeys(i for i in tuning_ids if isinstance(i, str))
                          if i in experiments]
    if validation_records:
        reference = validation_records[0]
        for record in validation_records[1:]:
            if record.get("scenarios") != reference.get("scenarios") or record.get("trial_keys") != reference.get("trial_keys"):
                warnings.append(f"验证实验 {record.get('experiment_id')} 的完整场景或 trial 身份不一致，不能作同数据比较。")
    holdout_records = []
    for label, identifier in _mapping(summary.get("holdout")).items():
        record = experiments.get(identifier) if isinstance(identifier, str) else None
        if record:
            holdout_records.append(record)
            if record.get("split") not in {"test", "midterm_holdout"}:
                warnings.append(f"holdout 标签 {label} 的实际 split 不是封存测试集；不能当测试收益。")
    validation_seeds = {s.get("seed") for identifier in tuning_ids if isinstance(identifier, str)
                        for s in experiments.get(identifier, {}).get("scenarios", [])
                        if isinstance(s, dict) and isinstance(s.get("seed"), int)}
    holdout_seeds = {s.get("seed") for record in holdout_records for s in record.get("scenarios", [])
                     if isinstance(s, dict) and isinstance(s.get("seed"), int)}
    if validation_seeds & holdout_seeds:
        warnings.append("验证与 holdout 实验 seed 存在重叠；不能宣称独立随机种子协议已满足。")
    keys = [r.get("trial_keys") for r in holdout_records]
    if keys and any(k is None for k in keys):
        warnings.append("部分 holdout 缺少 trial_keys；报告不能独立核实 bootstrap 配对身份。")
    elif keys and any(k != keys[0] for k in keys[1:]):
        warnings.append("holdout 的 trial_keys 顺序不相同；配对 CI 的身份对齐需单独核实。")
    if holdout_records:
        reference = holdout_records[0]
        for record in holdout_records[1:]:
            if record.get("scenarios") != reference.get("scenarios"):
                warnings.append(f"holdout {record.get('experiment_id')} 的完整场景配置不一致，不能仅凭 trial_keys 保证配对。")
        selected_id = _mapping(summary.get("holdout")).get("selected")
        selected = experiments.get(selected_id) if isinstance(selected_id, str) else None
        if selected and selected.get("method") != summary.get("final_method"):
            warnings.append("holdout-selected 方法与冻结的 final_method 不一致。")
    baseline_id = _mapping(summary.get("holdout")).get("nls")
    baseline = experiments.get(baseline_id) if isinstance(baseline_id, str) else None
    for label, result in _mapping(summary.get("paired_bootstrap")).items():
        result = _mapping(result)
        candidate_id = _mapping(summary.get("holdout")).get(label)
        candidate = experiments.get(candidate_id) if isinstance(candidate_id, str) else None
        if result.get("comparison") != "candidate_minus_baseline":
            warnings.append(f"CI {label} 比较方向不是 candidate_minus_baseline，禁止直接解释收益。")
        if baseline and candidate:
            if result.get("n_pairs") != len(candidate.get("trial_keys", [])):
                warnings.append(f"CI {label} 的配对数与实验 trial_keys 不一致。")
            for metric in ("cep90_m", "median_error_m", "failure_rate_over_30m"):
                point = _mapping(result.get(metric)).get("point_estimate")
                bm, cm = _mapping(baseline.get("metrics")).get(metric), _mapping(candidate.get("metrics")).get(metric)
                if all(isinstance(v, (int, float)) for v in (point, bm, cm)) and not math.isclose(point, cm - bm, abs_tol=1e-10):
                    warnings.append(f"CI {label}/{metric} 点估计与引用实验差值不一致。")
    if summary.get("status") not in {"completed", "complete", "success", "finished"}:
        warnings.append("运行状态不是明确的完成状态；本报告只是当前产物快照。")
    if not summary.get("frozen_at_utc"):
        warnings.append("未记录冻结时间；不能证明 holdout 在方案冻结后运行。")
    elif holdout_records:
        try:
            frozen = datetime.fromisoformat(str(summary["frozen_at_utc"]).replace("Z", "+00:00"))
            if frozen.tzinfo is None:
                raise ValueError("freeze timestamp has no timezone")
            for record in holdout_records:
                timestamp = _mapping(record.get("environment")).get("timestamp_utc")
                evaluated = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                if evaluated.tzinfo is None or evaluated < frozen:
                    warnings.append(f"holdout {record.get('experiment_id')} 时间早于冻结或无时区，不能确认冻结后评估。")
        except (ValueError, TypeError):
            warnings.append("冻结时间或 holdout 运行时间无法解析；无法核实冻结后评估顺序。")
    return summary, records, experiments, warnings


def _register_fonts():
    """Prefer embeddable TrueType CJK; CFF-based Noto TTC is not TTFont-safe."""
    latin = "Helvetica"
    latin_chars = set(range(32, 127))
    latin_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if latin_path.exists():
        try:
            font = TTFont("MidtermLatin", str(latin_path))
            pdfmetrics.registerFont(font)
            latin, latin_chars = font.fontName, set(font.face.charToGlyph)
        except (ValueError, OSError):
            pass
    candidates = [Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")]
    try:
        result = subprocess.run(
            ["fc-list", ":lang=zh", "file"], capture_output=True, text=True,
            timeout=5, check=False,
        )
        candidates.extend(Path(line.split(":", 1)[0]) for line in result.stdout.splitlines()
                          if line.split(":", 1)[0].lower().endswith((".ttf", ".ttc")))
    except (OSError, subprocess.TimeoutExpired):
        pass
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            font = TTFont("MidtermChinese", str(path))
            chars = set(font.face.charToGlyph)
            if not all(ord(ch) in chars for ch in "中文中期实验报告验证错误静态目标"):
                continue
            pdfmetrics.registerFont(font)
            return font.fontName, chars, latin, latin_chars, f"嵌入中文字体：{path.name}"
        except Exception:  # Unsupported TTC outlines should fall through to CID.
            continue
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    chars = set(range(0x4E00, 0xA000)) | set(range(0x3000, 0x3040)) | set(range(0xFF01, 0xFF5F))
    return "STSong-Light", chars, latin, latin_chars, (
        "中文字体回退：STSong-Light CID（非嵌入字体；须在交付环境复核渲染）。"
    )


class _Graph(Flowable):
    """Vector graph with Chinese labels rendered through the same Paragraphs."""
    def __init__(self, paragraph):
        super().__init__()
        self.width, self.height = _WIDTH, 112
        self.paragraph = paragraph

    def draw(self):
        c = self.canv
        nodes = ["读取验证结果", "生成单一提案", "白名单校验", "受限实验执行", "规则接受/拒绝", "持久化日志"]
        box_w, box_h = 145, 31
        positions = [(0, 77), (175, 77), (350, 77), (350, 16), (175, 16), (0, 16)]
        c.setStrokeColor(colors.HexColor("#345c78"))
        for label, (x, y) in zip(nodes, positions):
            c.setFillColor(colors.HexColor("#eef4f7"))
            c.roundRect(x, y, box_w, box_h, 4, fill=1, stroke=1)
            p = self.paragraph(label, "cell")
            _, h = p.wrap(box_w - 16, box_h)
            p.drawOn(c, x + 8, y + (box_h - h) / 2)
        for x1, y1, x2, y2 in [(145, 93, 175, 93), (320, 93, 350, 93),
                                (422, 77, 422, 47), (350, 32, 320, 32), (175, 32, 145, 32)]:
            c.line(x1, y1, x2, y2)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            c.line(x2, y2, x2 - 5 * ux + 3 * uy, y2 - 5 * uy - 3 * ux)
            c.line(x2, y2, x2 - 5 * ux - 3 * uy, y2 - 5 * uy + 3 * ux)


class _Page(KeepInFrame):
    """Keep explicit sections within eight pages, flag unusually dense inputs."""
    def drawOn(self, canv, x, y, _sW=0):
        super().drawOn(canv, x, y, _sW)
        if getattr(self, "_scale", 1.0) > 1.3:
            canv.saveState()
            canv.setFont("Helvetica", 7)
            canv.setFillColor(colors.HexColor("#9d341e"))
            canv.drawString(42, 37, "Dense evidence: page scaled; use source JSON/JSONL for full-size audit.")
            canv.restoreState()


def _comparison_plot(rows):
    """No data-dependent conclusion; numbered labels correspond to the table."""
    usable = [(i + 1, _mapping(record.get("metrics"))) for i, (_, _, record) in enumerate(rows)
              if record and all(isinstance(_mapping(record.get("metrics")).get(k), (int, float))
                                and math.isfinite(_mapping(record.get("metrics"))[k])
                                for k in ("cep90_m", "median_error_m"))]
    if not usable:
        return None
    # Import lazily: missing plotting dependencies should not destroy a PDF.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(7.2, 2.15))
    x = list(range(len(usable)))
    ax.bar([a - .18 for a in x], [m["cep90_m"] for _, m in usable], .36, label="CEP90")
    ax.bar([a + .18 for a in x], [m["median_error_m"] for _, m in usable], .36, label="Median")
    ax.set_xticks(x, [f"M{i}" for i, _ in usable])
    ax.set_ylabel("Error (m)")
    ax.set_title("Validation only - recorded methods")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    buffer.seek(0)
    return Image(buffer, width=_WIDTH, height=148)


def build_midterm_report(run_id: str, *, artifact_base: str | Path | None = None) -> Path:
    """Build artifacts/midterm/<run_id>/report/midterm_report_zh.pdf.

    Raises on absent/invalid summary or mismatched run ID. Optional missing
    experiments, log lines, holdout and CIs become visible evidence warnings.
    No environment mutation, API calls, experiment execution or Git operations.
    """
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must be a safe alphanumeric identifier (maximum 80 characters)")
    base = Path(artifact_base) if artifact_base is not None else ROOT / "artifacts" / "midterm"
    run = base.expanduser().resolve() / run_id
    summary, records, experiments, warnings = _load_evidence(run, run_id)
    cjk, cjk_chars, latin, latin_chars, font_note = _register_fonts()
    styles = {
        "body": ParagraphStyle("body", fontName=cjk, fontSize=9, leading=14, spaceAfter=6, wordWrap="CJK"),
        "title": ParagraphStyle("title", fontName=cjk, fontSize=23, leading=31, spaceAfter=16, textColor=colors.HexColor("#183f58"), wordWrap="CJK"),
        "h1": ParagraphStyle("h1", fontName=cjk, fontSize=17, leading=24, spaceAfter=12, textColor=colors.HexColor("#183f58"), wordWrap="CJK"),
        "h2": ParagraphStyle("h2", fontName=cjk, fontSize=11, leading=17, spaceBefore=6, spaceAfter=7, textColor=colors.HexColor("#345c78"), wordWrap="CJK"),
        "cell": ParagraphStyle("cell", fontName=cjk, fontSize=7.6, leading=11.5, wordWrap="CJK"),
        "small": ParagraphStyle("small", fontName=cjk, fontSize=7.5, leading=11, spaceAfter=5, wordWrap="CJK"),
        "header": ParagraphStyle("header", fontName=cjk, fontSize=7.6, leading=11.5, wordWrap="CJK", textColor=colors.white),
    }

    def p(value, style="body"):
        # External prose is text, never ReportLab markup. Split font runs so the
        # Droid CJK fallback's missing Latin glyphs cannot turn into black boxes.
        runs, current, active = [], [], None
        for ch in _text(value):
            code = ord(ch)
            if ch == "\n":
                if current:
                    runs.append(f'<font name="{active}">{escape("".join(current))}</font>')
                    current = []
                runs.append("<br/>")
                active = None
                continue
            if code < 32:
                ch = " "
                code = 32
            # Do not use Unicode super/subscripts, even if one installed font has them.
            if code > 0xFFFF or 0x2070 <= code <= 0x209F or code in (0xB2, 0xB3, 0xB9):
                ch, font = f"[U+{code:04X}]", latin
            elif code in latin_chars and (code < 0x3000 or code not in cjk_chars):
                font = latin
            elif code in cjk_chars:
                font = cjk
            elif code in latin_chars:
                font = latin
            else:
                ch, font = f"[U+{code:04X}]", latin
            if font != active and current:
                runs.append(f'<font name="{active}">{escape("".join(current))}</font>')
                current = []
            current.append(ch)
            active = font
        if current:
            runs.append(f'<font name="{active}">{escape("".join(current))}</font>')
        return Paragraph("".join(runs), styles[style])

    def table(headers, data, widths):
        flow = Table([[p(c, "header") for c in headers]] +
                     [[p(c, "cell") for c in row] for row in data],
                     colWidths=[_WIDTH * w / sum(widths) for w in widths], repeatRows=1,
                     hAlign="LEFT")
        flow.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24536e")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f5f7"), colors.white]),
            ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#c7d3da")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return flow

    def referenced(label, identifier):
        record = experiments.get(identifier) if isinstance(identifier, str) else None
        return (label, identifier, record)

    def method_text(record):
        if not record:
            return "产物缺失"
        method = _mapping(record.get("method"))
        keys = ("name", "loss_scale_m", "reliability_threshold", "max_irls_iterations", "use_temporal_weight")
        known = "；".join(f"{short}={_text(method.get(k))}" for short, k in zip(("name", "s", "threshold", "IRLS", "temporal"), keys))
        extras = {k: v for k, v in method.items() if k not in keys}
        return known + ("；其他=" + _text(extras) if extras else "")

    validations = [referenced("基线验证", summary.get("baseline_validation"))]
    for label, identifier in _mapping(summary.get("comparisons")).items():
        if identifier == summary.get("baseline_validation"):
            validations[0] = referenced("基线验证 / " + label, identifier)
        else:
            validations.append(referenced(label, identifier))
    validations.append(referenced("最终验证", summary.get("final_validation")))
    holdouts = [referenced(label, identifier) for label, identifier in _mapping(summary.get("holdout")).items()]
    final = experiments.get(summary.get("final_validation")) if isinstance(summary.get("final_validation"), str) else None
    final_metrics = _mapping(final.get("metrics")) if final else {}
    policy = summary.get("policy")
    traces = [(r, t) for r in records for t in r.get("model_traces", []) if isinstance(t, dict)]
    logged_errors = sum(bool(r.get("error") or r.get("status") == "error") for r in records)
    evaluated = sum(r.get("status") == "evaluated" for r in records)
    blocked_attempts = sum(a.get("status") == "invalid" for r in records
                           if r.get("error") or r.get("status") == "error"
                           for a in r.get("proposal_attempts", []) if isinstance(a, dict))
    verified_responses = len({t.get("response_id") for _, t in traces
                              if isinstance(t.get("response_id"), str) and t.get("raw_response")})
    round_note = (f"实际可读日志：{evaluated} 轮候选评估，{logged_errors} 轮错误；错误轮次共 {blocked_attempts} 个无效提案尝试被阻断。"
                  f"日志保留 {verified_responses} 个带独立 response_id 和原始内容的模型响应；响应数不等于实验轮数。")
    pages = []

    # 1. Scope and evidence-derived executive status, not a predicted result.
    pages.append([
        p("BLE 众包静态定位\n实验智能体中期报告", "title"),
        p("课程任务：开发智能体，用于应用数学问题研究与迭代实验。", "h2"),
        p("选题来源：《HarmonyOS难题发布-第七期》难题 1，基于非线性优化的众包离线高精度位置聚合。本项目研究其中可复现的二维静态目标定位子问题：从帮助设备报告位置与 BLE RSSI 估计离线设备位置。"),
        p("范围声明：每次试验中的目标位置保持静态；帮助设备坐标漂移、重复距离变化不等于目标移动。未实现目标轨迹或移动目标分类。合成结果不能作为满足 Huawei 生产 KPI 的证据。"),
        table(["运行字段", "summary / 产物实际记录"], [
            ("run_id / status", f"{run_id} / {_text(summary.get('status'))}"),
            ("策略 policy", _text(policy)),
            ("冻结时间 UTC", _text(summary.get("frozen_at_utc"))),
            ("迭代记账 / API 已评估 / 错误轮数", " / ".join(_text(summary.get(k)) for k in
                ("completed_iterations", "api_evaluated_iterations", "error_iterations"))),
            ("可读日志 / 模型 trace / 日志错误", f"{len(records)} / {len(traces)} / {logged_errors}"),
            ("最终方法（完整配置）", _text(summary.get("final_method"))),
            ("最终验证 CEP90 / 中位误差", f"{_number(final_metrics.get('cep90_m'))} m / {_number(final_metrics.get('median_error_m'))} m"),
            ("holdout 结果引用数", str(len(holdouts))),
        ], [1, 2.7]),
        Spacer(1, 12),
        p("解读规则", "h2"),
        p("最终方法以 summary 为准，验证指标以其引用的实验 JSON 为准。没有 API trace 的轮次不标称成功模型推理；错误轮次不冒充已评估候选。拒绝提案仍是有效过程证据，但不能算作性能改进。"),
        p(round_note),
        p("研究贡献是既有鲁棒方法与受限实验闭环的实现和比较，不是新定位算法。模型理由是待检验假设，不是数学或因果证明。", "small"),
        p("本报告仅读取本 run 的 summary.json、iterations/agent_iterations.jsonl 和 experiments/*.json；不调用模型，不运行实验，不读取凭据。", "small"),
    ])

    # 2. Mathematical implementation and repairs verified from current source.
    pages.append([
        p("02  数学模型与实现修复", "h1"),
        p("观测与目标", "h2"),
        p("未知静态二维位置 x；帮助设备报告位置 l_i；RSSI r_i；估计距离 d_i = f(r_i)。残差 e_i(x) = norm(x - l_i) - d_i。评价误差为 norm(x_hat - x_true)，真值只在仿真评价端使用。"),
        p("对数路径损耗：r_i = A - 10*n*log10(d_i) + b；反演 d_i = 10^((A-r_i)/(10*n))。A、n 为 calibration.py 中的显式仿真假设，不是实测标定。校准偏差 b 以 dBm 施加一次，当前距离与时间窗口使用同一转换。"),
        p("优化问题：min_x sum_i rho(w_i * e_i(x)^2 / s^2)。SciPy 实际接收 sqrt(w_i)*e_i(x) 与 f_scale=s。鲁棒损失作用于加权残差，不应误写为 sum_i w_i*rho(e_i(x)^2)。线性损失时等价于加权最小二乘。"),
        table(["方法", "实际机制与边界"], [
            ("nls / weighted_nls", "平方损失；weighted_nls 使用 metadata 权重。时间权重由配置显式开启，不是隐藏的基线优势。所有方法均从报告位置均值初始化。"),
            ("Huber / Cauchy", "对加权残差采用鲁棒损失，尺度控制抑制大残差的程度。改善与否由实验决定，不能先验保证。"),
            ("adaptive", "交替执行 Huber 坐标拟合与可靠性更新；w = metadata * residual_weight，开启时再乘 temporal_weight。阈值以下权重降至 0.01。"),
            ("权重公式", "residual_weight = clip(1/(1+(abs(e)/s)^2), 0.02, 1)；时间不稳定度 u=std(d_windows)/s，temporal_weight=clip(1/(1+u^2), 0.03, 1)。"),
        ], [1, 3.1]),
        p("已落实的正确性修复（实现事实，不是测试通过声明）", "h2"),
        p("1. 求解入口校验维度、有限值、正距离、参数类型与范围；未知方法拒绝执行。\n2. 所有方法使用报告坐标均值初始化，切断模拟真值与污染标签的求解器入口。\n3. adaptive 最后权重更新后追加一次固定权重拟合，返回坐标与返回权重一致。\n4. 时间权重关闭时，不稳定度为零；开启时高分代表更不稳定，不是概率。\n5. adaptive 的 iterations 是 IRLS 权重更新数；其他方法是 SciPy 函数评估数，两者不直接比较。"),
        p("confidence_radius_m 是残差与几何条件构成的启发式量，不是经过覆盖率校准的置信区域。收敛率代表数值求解器 success，不等于正确定位率。", "small"),
    ])

    protocol_rows = [(key, _excerpt(value, 600)) for key, value in _flatten(_mapping(summary.get("protocol")))]
    pages.append([
        p("03  Agent 图、角色与验证协议", "h1"), _Graph(p),
        p("提案失败分支直接持久化错误，不运行候选实验；API 请求失败不暗中切换 scripted。图由 LangGraph 执行。", "small"),
        table(["角色", "允许 / 不允许"], [
            ("模型或策略", "读取验证指标、切片、当前配置与历史，提出一个白名单字段改动。没有 shell、任意代码、真值数组或 holdout 工具。api、diagnostic 与 scripted 分别为模型、结果驱动本地规则与固定计划，不混称。"),
            ("校验与执行器", "Pydantic 严格 schema + 白名单；拒绝无效类型、无变化、无效参数或重复完整配置。执行器仅允许 validation / midterm_validation。"),
            ("确定性决策", "要求 CEP90 严格降低；总体中位误差不恶化超过 10%；>30m 失败率不增加；收敛率下降不超过 5 个百分点；clean 切片身份相同且 CEP90 恶化不超过 max(10%, 0.25m)。这些是当前代码规则，不是显著性检验。"),
        ], [1, 3.1]),
        p("元数据独立于污染标签", "h2"),
        p("observable metadata 的定位精度和观测年龄代理来自独立随机流，与 outlier/mobile 标签及漂移独立；仍是合成代理而非实测质量。oracle 模式包含污染信息，仅能作理想化上界，Agent 调参禁止使用。移动帮助设备标签只留在评价端。"),
        p("本 run 的协议（summary.protocol 原值）", "h2"),
        table(["字段", "实际值"], protocol_rows or [("protocol", "未记录；不能确认具体划分和预算")], [1.3, 2.8]),
        p("新验证集用于诊断、基线消融与方案选择；holdout 必须在冻结后仅作最终比较，不能用于下一轮调参。不同 split 的实际种子、场景与配置应由实验 JSON 核对，不以文件名替代独立性证据。", "small"),
    ])

    def metric_table(rows):
        data = []
        for index, (label, identifier, record) in enumerate(rows, 1):
            m = _mapping(record.get("metrics")) if record else {}
            data.append((f"M{index} {label}\n{_text(identifier)}\n{_text(record.get('split')) if record else '产物缺失'}",
                         _text(m.get("n_trials")), _number(m.get("mean_error_m")),
                         _number(m.get("median_error_m")), _number(m.get("cep90_m")),
                         _number(m.get("failure_rate_over_30m"), percent=True),
                         _number(m.get("convergence_rate"), percent=True),
                         _number(m.get("mean_runtime_ms"))))
        return table(["比较 / ID / split", "试验数", "均值 m", "中位 m", "CEP90 m", ">30m", "收敛率", "均时 ms"],
                     data or [("未提供", *([_MISSING] * 7))], [3.1, .6, .8, .8, .8, .8, .8, .8])

    page = [p("04  验证基线与消融：仅列实际实验", "h1"),
            p("下面标签与 ID 完全来自 summary.baseline_validation、comparisons 和 final_validation。未出现的消融不补造；重复 ID 表示同一产物，不视为独立试验。数值越低通常误差越小，但跨 split、样本或场景配置不能直接归因。"),
            metric_table(validations)]
    try:
        plot = _comparison_plot(validations)
        if plot:
            page += [Spacer(1, 8), plot]
    except (ImportError, ValueError, RuntimeError):
        page.append(p("图表未生成；上表保留实际指标。", "small"))
    method_rows = [(label, method_text(record)) for label, _, record in validations]
    page += [p("比较方法完整配置", "h2"), table(["标签", "name / s=loss_scale_m / threshold=reliability_threshold / IRLS=max_irls_iterations / temporal=use_temporal_weight"], method_rows, [1, 3.1])]
    slices = final.get("slices", []) if final else []
    slices = [s for s in slices if isinstance(s, dict)]
    worst = sorted(slices, key=lambda s: float(s.get("cep90_m", -1)) if isinstance(s.get("cep90_m"), (int, float)) else -1, reverse=True)[:4]
    page += [p(f"最终验证最差切片（前 {len(worst)} / {len(slices)} 个；完整切片见实验 JSON）", "h2"),
             table(["场景 / 布局 / seed / clean", "CEP90 m", "中位 m", ">30m"], [
                 (f"{_text(s.get('scenario_name', s.get('name')))} / {_text(s.get('layout'))} / {_text(s.get('seed'))} / {_text(s.get('clean'))}",
                  _number(s.get("cep90_m")), _number(s.get("median_error_m")), _number(s.get("failure_rate_over_30m"), percent=True))
                 for s in worst] or [("无切片记录", _MISSING, _MISSING, _MISSING)], [2.5, .8, .8, .8]),
             p("clean 表示无异常、帮助设备移动和校准偏差，仍允许常规距离和坐标噪声。切片按包含 seed、metadata_mode 与噪声等完整条件分组，不把同布局不同场景错误合并。", "small")]
    pages.append(page)

    # 5-6. Every readable round appears, including nullable proposals and failures.
    midpoint = (len(records) + 1) // 2
    for part, subset in enumerate((records[:midpoint], records[midpoint:]), 1):
        section = [p(f"0{4 + part}  逐轮提案与决策（{part}/2）", "h1")]
        if not subset:
            section.append(p("此部分没有可读迭代记录。无日志不能推断模型成功，也不能将缺失候选视为拒绝或采纳。"))
        for row in subset:
            decision = _mapping(row.get("decision"))
            proposal = _mapping(row.get("proposal"))
            current = _mapping(decision.get("current_metrics"))
            candidate = _mapping(decision.get("candidate_metrics"))
            error = row.get("error")
            blocked = bool(error or row.get("status") == "error") and not row.get("candidate_experiment_id")
            status = "阻断 / 未运行候选实验" if blocked else "错误 / 未完成评估" if error or row.get("status") == "error" else (
                "采纳" if decision.get("accepted") is True else "拒绝" if decision.get("accepted") is False else "决策未记录")
            section += [p(f"轮次 {_text(row.get('iteration_index'))} | {_text(row.get('policy'))} | {status}", "h2"),
                        p(f"输入 ID：{_text(row.get('input_experiment_id'))}\n候选 ID：{_text(row.get('candidate_experiment_id'))}", "small"),
                        p(f"实际改动：{_text(proposal.get('change')) if proposal else '无有效提案（proposal 为空）'}\n模型假设（待检验）：{_excerpt(proposal.get('hypothesis'), 300)}\n模型理由（非证明）：{_excerpt(proposal.get('rationale'), 400)}", "small"),
                        p(f"CEP90：{_number(current.get('cep90_m'))} -> {_number(candidate.get('cep90_m'))} m；中位：{_number(current.get('median_error_m'))} -> {_number(candidate.get('median_error_m'))} m；失败率：{_number(current.get('failure_rate_over_30m'), percent=True)} -> {_number(candidate.get('failure_rate_over_30m'), percent=True)}。", "small"),
                        p("决策原因：" + _excerpt(decision.get("reason"), 700), "small")]
            if error:
                section.append(p("错误记录：" + _excerpt(error, 700), "small"))
            attempts = row.get("proposal_attempts", [])
            if attempts:
                section.append(p("提案尝试：" + "; ".join(
                    f"attempt={_text(a.get('attempt'))}, status={_text(a.get('status'))}, error_type={_text(a.get('error_type'))}"
                    for a in attempts if isinstance(a, dict)), "small"))
            if blocked:
                invalid = sum(a.get("status") == "invalid" for a in attempts if isinstance(a, dict))
                section.append(p(f"该轮 {invalid} 个无效提案被阻断；没有候选实验，不计为真实评估或性能改善。", "small"))
            row_traces = [t for t in row.get("model_traces", []) if isinstance(t, dict)]
            if row_traces:
                for trace in row_traces:
                    usage = _mapping(trace.get("usage"))
                    tokens = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage}
                    section.append(p(
                        f"trace：model={_excerpt(trace.get('model'), 120)}；response_id={_excerpt(trace.get('response_id'), 160)}；prompt_version={_text(trace.get('prompt_version'))}；inference_seconds={_number(trace.get('inference_seconds'))}；usage（token 摘要）={_text(tokens or usage)}", "small"))
            else:
                section.append(p("无模型 trace；不能据此声称该轮成功 API 推理。", "small"))
            candidate_slices = [s for s in decision.get("candidate_slices", []) if isinstance(s, dict)]
            section.append(p(f"候选切片记录数：{len(candidate_slices)}；完整配置、切片与模型响应审计保留在源日志（本报告不复制 prompt 或 raw_response）。", "small"))
        section.append(p("保持当前配置或拒绝提案是规则执行结果，不代表 LLM 优于网格或随机搜索；本 run 若未做等预算搜索对照，就不作该结论。", "small"))
        pages.append(section)

    # 7. Holdout and already computed paired CIs, never recomputed from tuning data.
    section = [p("07  冻结后 holdout 与配对置信区间", "h1"),
               p("冻结时间（summary）：" + _text(summary.get("frozen_at_utc")), "small")]
    def compact_table(headers, data, widths):
        flow = table(headers, data, widths)
        flow.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 2),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return flow
    if holdouts:
        holdout_table = metric_table(holdouts)
        holdout_table.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 2),
                                          ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        section += [holdout_table, p("实际 holdout 数据身份与运行环境", "h2")]
        descriptor_rows = []
        for label, _, record in holdouts:
            if not record:
                descriptor_rows.append((label, "实验产物缺失"))
            else:
                descriptor_rows.append((label, _text(_mapping(record.get("environment")).get("timestamp_utc"))))
        reference = next((record for _, _, record in holdouts if record), None)
        if reference:
            scenarios = [s for s in reference.get("scenarios", []) if isinstance(s, dict)]
            section.append(p(
                f"共同描述（需结合审计一致性提示）：scenario 数={len(scenarios)}；"
                f"seed={_text(sorted({s.get('seed') for s in scenarios if isinstance(s.get('seed'), int)}))}；"
                f"metadata_mode={_text(sorted({str(s.get('metadata_mode', '未记录')) for s in scenarios}))}。"
                "完整场景与 trial_keys 见源实验；下表保留每个实际运行时间。", "small"))
        section.append(compact_table(["标签", "timestamp_utc"], descriptor_rows, [1, 3.1]))
    else:
        section.append(p("未提供 holdout 引用：冻结后测试未做、未完成或尚未写入 summary；不能报告测试收益。验证集结果不能替代测试。"))
    section += [p("已记录的普通配对 bootstrap", "h2"),
                p("差值方向为 candidate - baseline，负值表示候选更低。CI95 是百分位区间。对同场景同 trial 对齐的两方法误差，用相同索引进行普通配对重采样；仅描述固定场景混合的试验不确定性，不是分层/层级 bootstrap，不覆盖新的场景分布，也不是外部真实验证。区间包含零时不将点估计直接描述为确定收益。")]
    ci_rows = []
    bootstraps = _mapping(summary.get("paired_bootstrap"))
    for label, result in bootstraps.items():
        result = _mapping(result)
        cells = [str(label)]
        for key in ("cep90_m", "median_error_m", "failure_rate_over_30m"):
            item = _mapping(result.get(key))
            ci = item.get("ci95")
            def format_delta(value):
                if key == "failure_rate_over_30m" and isinstance(value, (int, float)) and math.isfinite(value):
                    return f"{value:.6f}"
                return _number(value)
            interval = f"[{format_delta(ci[0])}, {format_delta(ci[1])}]" if isinstance(ci, (list, tuple)) and len(ci) == 2 else "未计算 / 无有效区间"
            cells.append(f"{format_delta(item.get('point_estimate'))}\n{interval}")
        ci_rows.append(cells)
    section.append(compact_table(["比较标签", "CEP90 差 m / CI95", "中位差 m / CI95", "失败率差（比例）/ CI95"], ci_rows or
                         [("未做 / 失败 / 未记录", *([_MISSING] * 3))], [1, 1.5, 1.5, 1.5]))
    selected_ci = _mapping(bootstraps.get("selected"))
    failure_ci = _mapping(selected_ci.get("failure_rate_over_30m")).get("ci95")
    if isinstance(failure_ci, (list, tuple)) and len(failure_ci) == 2 and all(isinstance(v, (int, float)) for v in failure_ci):
        if failure_ci[0] <= 0 <= failure_ci[1]:
            section.append(p("selected 的失败率差 CI95 跨零：即使点估计较低，也不能确认失败率改善；比例差乘 100 才是百分点。", "small"))
    metadata_rows = []
    for label, result in bootstraps.items():
        result = _mapping(result)
        metadata_rows.append((str(label), _text(result.get("n_pairs")), _text(result.get("seed")),
                              _text(result.get("repeats")), _text(result.get("comparison"))))
    if metadata_rows:
        if all(row[1:] == metadata_rows[0][1:] for row in metadata_rows):
            section.append(p("全部已列 CI 共同元数据：n_pairs=" + metadata_rows[0][1] +
                             "；seed=" + metadata_rows[0][2] + "；repeats=" + metadata_rows[0][3] +
                             "；comparison=" + metadata_rows[0][4] + "。", "small"))
        else:
            section.append(compact_table(["CI 标签", "配对数", "seed", "repeats", "方向"], metadata_rows, [1.5, .7, .8, .8, 2.1]))
    section.append(p("完整 CI 元数据及 bootstrap_scope 保留在 summary；本表的区间不覆盖新的场景分布。", "small"))
    section.append(p(">30m 是本研究的错误阈值，不是 Huawei 官方验收。mean_runtime_ms 仅是 localize 求解计时，不含仿真、API、图编排或报告生成；模型耗时在逐轮 trace 中单列。", "small"))
    pages.append(section)

    # 8. Audit, testing honesty, limitations, and report-only reproduction.
    environment = _mapping(final.get("environment")) if final else {}
    audit = warnings or ["本次读取未发现缺失引用或日志行解析错误；这不是实验真实性或测试通过的认证。"]
    verification = run / "verification" / "release-verification.json"
    test_note = "报告生成器不运行 pytest，也不推断测试已通过。若未另行提供测试日志，测试状态为未在本报告中验证。缺失 CI 或错误迭代按产物实情展示。"
    if verification.exists():
        checked = _read_object(verification)
        test_note = "独立交付核验记录（不改写原实验）：" + _text(checked.get("test_result")) + "；" + _text(checked.get("verified_at_utc")) + "。完整终端日志和源码摘要见 verification/。这是交付时测试，不追溯认证最初运行；本生成器不运行测试。"
    section = [p("08  局限、审计与复现", "h1"),
               p("证据完整性与测试状态", "h2"),
               p("\n".join(f"- {message}" for message in audit), "small"),
               p(test_note),
               p("运行环境（最终验证实验 environment 原值）", "h2"),
               p(_excerpt(environment, 1800) if environment else "未记录运行环境。", "small"),
               p(font_note, "small"),
               p("研究局限与下一步", "h2"),
               p("1. 当前是简化二维静态仿真，缺少真实 BLE 传播、设备异构、楼层地图与时间延迟；metadata 是独立合成代理。\n2. 帮助设备不稳定度不等于目标运动；需要目标轨迹、定义明确的运动标签与实测验证。\n3. 验证集反复选择有过拟合风险；holdout 应保持封存，查看后不得再用于本轮选择。\n4. bootstrap 只针对固定场景混合；应进一步按场景报告并研究新场景与真实数据的泛化。\n5. 如需评价模型选参价值，补做等预算网格/随机/本地诊断对照、多个运行与成本统计；在此之前不宣称 LLM 更优。\n6. 用授权脱敏真实数据重新标定、评估覆盖率与官方 KPI；本中期不承诺满足 Huawei KPI。"),
               p("登记边界：新 run 应在实验前完整保存配置 snapshot 与生成器 hash。本报告不认证事前登记链；已有 run 后补快照只能称事后核验，不能把当前源码 hash 追溯为历史运行身份。冻结后恢复只应补缺失，不改选择、不重跑已有证据。", "small"),
               p("本研究不宣称发明新定位算法；模型理由仅是待检验假设，不是机制证明。", "small"),
               p("复现已有产物的报告（不会跑实验）", "h2"),
               p("在项目根目录、已安装的环境中执行：\npython scripts/build_midterm_report.py --run-id " + run_id, "small"),
               p("输入：artifacts/midterm/" + run_id + "/summary.json；iterations/agent_iterations.jsonl；experiments/*.json。\n输出：artifacts/midterm/" + run_id + "/report/midterm_report_zh.pdf。", "small"),
               p("完整实验复现应使用本 run 协议记录及对应运行器，不以本生成器重跑实验；保留随机种子、完整配置、trial_keys、原始误差、冻结时间、模型 trace 和错误日志。演示讲稿见 docs/midterm_presentation.md，所有数字以本报告与源 JSON 为准。", "small")]
    pages.append(section)

    report_dir = run / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "midterm_report_zh.pdf"
    story = []
    for index, content in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.append(_Page(_WIDTH, _HEIGHT, content, mode="shrink", hAlign="LEFT", vAlign="TOP"))

    def chrome(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c7d3da"))
        canvas.line(42, A4[1] - 34, A4[0] - 42, A4[1] - 34)
        canvas.setFillColor(colors.HexColor("#345c78"))
        canvas.setFont(latin, 7)
        canvas.drawString(42, A4[1] - 25, "BLE | STATIC TARGET | MIDTERM EVIDENCE REPORT")
        canvas.drawString(42, 23, "Run: " + run_id)
        canvas.drawRightString(A4[0] - 42, 23, f"{doc.page} / {len(pages)}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=48, bottomMargin=42, title="BLE 众包静态定位实验智能体中期报告",
                            author="BLE localization experiment agent", pageCompression=1)
    doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
    return output
