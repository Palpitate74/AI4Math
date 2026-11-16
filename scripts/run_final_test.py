"""Deprecated v1 entry point; prevents selection from historical shared state."""
if __name__ == "__main__":
    raise SystemExit("Deprecated: use python scripts/run_midterm.py --run-id YOUR_RUN --policy api --iterations 5 --finalize. Selection freezes before isolated holdout evaluation.")
