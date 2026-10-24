"""ClauseLens chat UI. Runs locally (`streamlit run`) and as a Databricks App."""

from __future__ import annotations

import streamlit as st

from clauselens.config import get_settings
from clauselens.eval.scoring import (
    THRESHOLDS,
    expected_label,
    load_questions,
    load_results,
    run_one,
    save_results,
    setup_label,
)
from clauselens.factory import build_agent
from clauselens.models import CoverageAnswer, Verdict
from clauselens.parsing.loader import load_catalog

VERDICT_STYLE = {
    Verdict.COVERED: ("Covered", "green"),
    Verdict.NOT_COVERED: ("Not covered", "red"),
    Verdict.DEPENDS: ("Depends on the facts", "orange"),
    Verdict.CANNOT_DETERMINE: ("Can't determine from the wording", "gray"),
}

EXAMPLES = [
    "Is water that backs up through a floor drain covered?",
    "What does \"occurrence\" mean?",
    "What is the special limit for jewelry stolen from my home?",
    "Is damage from an earthquake covered?",
]


def _md(text: str) -> str:
    """Escape dollar signs so Streamlit doesn't render "$10,000 ... $500" as a math formula."""
    return text.replace("$", "\\$")


@st.cache_resource
def _agent():
    return build_agent()


@st.cache_data
def _catalog() -> list[dict]:
    return [entry for entry, _ in load_catalog(get_settings().forms_dir)]


def _render(answer: CoverageAnswer) -> None:
    label, color = VERDICT_STYLE[answer.verdict]
    st.markdown(f"**:{color}[{label}]**")
    if answer.base_verdict and answer.base_verdict != answer.verdict:
        base_label, _ = VERDICT_STYLE[answer.base_verdict]
        st.caption(f"Base form alone: {base_label}. Changed by an endorsement.")
    st.markdown(_md(answer.explanation))

    if answer.endorsement_effects:
        st.markdown("**Endorsement effects**")
        for effect in answer.endorsement_effects:
            summary = _md(effect.summary)
            st.markdown(f"- `{effect.endorsement_form}` *{effect.action.value}*: {summary}")
    if answer.citations:
        with st.expander(f"Cited clauses ({len(answer.citations)})"):
            for c in answer.citations:
                st.markdown(f"**[{c.ref}] {c.label()}**")
                st.caption(_md(c.text))
    if answer.definitions_used:
        with st.expander("Defined terms"):
            for term, text in answer.definitions_used.items():
                st.markdown(f"**{term}**: {_md(text)}")


METRIC_LABELS = {
    "verdict_match": "Correct answer",
    "citation_recall": "Right clause cited",
    "refusal_correct": "Refuses off-topic",
    "endorsement_detected": "Spots add-on changes",
}
VERDICT_WORDS = {"covered": "Covered", "not_covered": "Not covered", "depends": "Depends",
                 "cannot_determine": "Can't determine"}


def _clause_cell(recall: float | None) -> str:
    if recall is None:
        return "—"
    return "Yes" if recall >= 1.0 else ("No" if recall == 0 else f"{recall:.0%}")


def _results_table(rows: list[dict]) -> None:
    st.dataframe(
        [{"#": i, "Question": r["question"], "Policy": r["setup"], "Expected": r["expected"],
          "ClauseLens said": VERDICT_WORDS.get(r["verdict"], r["verdict"]),
          "Right clause": _clause_cell(r["citation_recall"]),
          "Result": "✅ Pass" if r["passed"] else "❌ Fail"}
         for i, r in enumerate(rows, 1)],
        hide_index=True, width="stretch")


def test_results_page() -> None:
    st.title("Test results")
    st.caption("14 questions with known correct answers, run through ClauseLens to check it "
               "answers correctly and cites the right clause.")
    questions = load_questions()
    results = load_results()

    if results:
        rows = results["rows"]
        passed = sum(r["passed"] for r in rows)
        st.markdown(f"### {passed} of {len(rows)} passed")
        st.caption(f"Last run: {results['run_at']} · model: {results['model']}")
        cols = st.columns(len(results["summary"]))
        for col, (name, value) in zip(cols, results["summary"].items(), strict=True):
            col.metric(METRIC_LABELS.get(name, name), f"{value:.0%}",
                       help=f"Minimum required: {THRESHOLDS[name]:.0%}")
        _results_table(rows)
        with st.expander("See each answer"):
            for i, r in enumerate(rows, 1):
                st.markdown(f"**{i}. {r['question']}** ({r['setup']})")
                st.caption(_md(r["answer"]))
    else:
        st.info("No test run yet. Click the button below, or run "
                "`python -m clauselens.eval.run` in a terminal.")
        st.dataframe([{"#": i, "Question": q["question"], "Policy": setup_label(q),
                       "Expected": expected_label(q)} for i, q in enumerate(questions, 1)],
                     hide_index=True, width="stretch")

    st.divider()
    if st.button(f"Run all {len(questions)} questions now", type="primary"):
        progress = st.progress(0.0, text="Starting…")
        rows = []
        for i, q in enumerate(questions, 1):
            progress.progress((i - 1) / len(questions), text=f"Question {i} of {len(questions)}: "
                                                            f"{q['question']}")
            rows.append(run_one(_agent(), q))
        progress.progress(1.0, text="Done")
        save_results(rows, model=get_settings().groq_model)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="ClauseLens", page_icon="🔎", layout="centered")
    page = st.sidebar.radio("View", ["Ask ClauseLens", "Test results"], horizontal=True)
    if page == "Test results":
        test_results_page()
        return
    st.title("ClauseLens")
    st.caption("Clause-cited answers about insurance policy wording. "
               "Informational only; not a coverage determination.")

    catalog = _catalog()
    base_forms = [e["form_number"] for e in catalog if not e.get("endorsement")]
    endorsements = [e["form_number"] for e in catalog if e.get("endorsement")]
    titles = {e["form_number"]: e.get("title", "") for e in catalog}

    with st.sidebar:
        st.header("Policy")
        forms = st.multiselect("Base form", base_forms, default=base_forms[:1],
                               format_func=lambda f: f"{f} — {titles[f]}")
        attached = st.multiselect("Attached endorsements", endorsements,
                                  format_func=lambda f: f"{f} — {titles[f]}")
        st.divider()
        st.markdown("**Try asking**")
        for example in EXAMPLES:
            if st.button(example, width="stretch"):
                st.session_state["pending"] = example

    history = st.session_state.setdefault("history", [])
    for question, answer in history:
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            _render(answer)

    question = st.chat_input("Ask about coverage, a definition or a limit")
    question = question or st.session_state.pop("pending", None)
    if not question:
        return
    if not forms:
        st.warning("Pick at least one base form in the sidebar.")
        return
    st.chat_message("user").write(question)
    with st.chat_message("assistant"), st.spinner("Reading the policy…"):
        answer = _agent().ask(question, forms=forms, endorsements=attached)
        _render(answer)
    history.append((question, answer))


main()
