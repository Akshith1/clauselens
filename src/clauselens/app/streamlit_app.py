"""ClauseLens chat UI. Runs locally (`streamlit run`) and as a Databricks App."""

from __future__ import annotations

import streamlit as st

from clauselens.config import get_settings
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


def main() -> None:
    st.set_page_config(page_title="ClauseLens", page_icon="🔎", layout="centered")
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
            if st.button(example, use_container_width=True):
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
