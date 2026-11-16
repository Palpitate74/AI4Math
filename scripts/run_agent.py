"""Deprecated v1 entry point; prevents mixing historical and v2 evidence."""
if __name__ == "__main__":
    raise SystemExit("Deprecated: use python scripts/run_midterm.py --run-id NEW_ID --policy api --iterations 5 [--finalize]. Historical artifacts remain read-only.")
