"""Run the question set through the agent and gate on thresholds.

    python -m clauselens.eval.run              # plain run, prints a table, exits 1 below threshold
    python -m clauselens.eval.run --mlflow     # also log a run with MLflow GenAI evaluation
"""

from __future__ import annotations

import argparse
import sys

from clauselens.config import get_settings
from clauselens.eval.scoring import (
    SCORERS,
    THRESHOLDS,
    answer_to_dict,
    load_questions,
    run_one,
    save_results,
    summarize,
)
from clauselens.factory import build_agent


def run_plain(agent, questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        row = run_one(agent, q)
        rows.append(row)
        print(f"{q['id']:<22} {row['verdict']:<17} "
              + "  ".join(f"{k}={row[k]}" for k in SCORERS if row[k] is not None))
    return rows


def run_mlflow(agent, questions: list[dict]) -> None:
    import mlflow
    from mlflow.genai.scorers import scorer

    def predict(question: str, forms: list[str], endorsements: list[str]) -> dict:
        return answer_to_dict(agent.ask(question, forms, endorsements))

    def wrap(name, fn):
        @scorer(name=name)
        def _score(outputs, expectations):
            return fn(outputs, expectations)
        return _score

    data = [{"inputs": {"question": q["question"], "forms": q["forms"],
                        "endorsements": q["endorsements"]},
             "expectations": {k: v for k, v in q.items()
                              if k.startswith("expect")}} for q in questions]
    mlflow.set_experiment("/Shared/clauselens-eval" if mlflow.get_tracking_uri()
                          .startswith("databricks") else "clauselens-eval")
    mlflow.genai.evaluate(data=data, predict_fn=predict,
                          scorers=[wrap(n, f) for n, f in SCORERS.items()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow", action="store_true", help="also log an MLflow evaluation")
    args = parser.parse_args()

    agent, questions = build_agent(), load_questions()
    rows = run_plain(agent, questions)
    summary = summarize(rows)
    saved = save_results(rows, model=get_settings().groq_model)
    print(f"\nResults saved to {saved} (shown on the app's Test results page)")
    print("\nSummary")
    failed = False
    for name, value in summary.items():
        ok = value >= THRESHOLDS[name]
        failed |= not ok
        status = "ok" if ok else "FAIL"
        print(f"  {name:<22} {value:.2f}  (threshold {THRESHOLDS[name]:.2f}) {status}")
    if args.mlflow:
        run_mlflow(agent, questions)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
