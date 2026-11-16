"""Generate a Chinese midterm PDF from an existing isolated run only."""
from __future__ import annotations

import argparse
import sys

from ble_agent.midterm_report import build_midterm_report


def main() -> int:
    parser = argparse.ArgumentParser(description="从已有 run 产物生成中文中期 PDF（不运行实验）")
    parser.add_argument("--run-id", required=True, help="artifacts/midterm 下的运行 ID")
    args = parser.parse_args()
    try:
        output = build_midterm_report(args.run_id)
    except (OSError, ValueError) as exc:
        print(f"报告生成失败：{exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
