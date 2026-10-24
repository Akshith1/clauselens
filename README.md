# ClauseLens

**Clause-cited coverage answers for insurance policies, built on Azure Databricks.**

ClauseLens answers coverage questions such as *"Is sewer backup covered under this homeowners
policy with endorsement HO 04 95?"*. It answers from the policy text itself, cites the
exact form, section, and clause, and flags when an endorsement changes what the base
form says. If the documents don't support an answer, it says so instead of guessing.

> Author: Rajesh Vasa. Built after completing training in agentic AI patterns on
> Databricks and Azure; ClauseLens is an independent project with its own domain,
> design, and implementation.

## What makes it different

| Concern | ClauseLens approach |
|---|---|
| Document structure | Parses policies by their legal structure: declarations, insuring agreement, definitions, exclusions, conditions, endorsements. No fixed-size chunks. |
| Citations | Every claim cites `form / section / clause`, e.g. `HO 00 03, Section I – Exclusions, A.3`. |
| Endorsements | An override check compares the base-form answer with attached endorsements and reports when an endorsement adds, removes, or limits coverage. |
| Defined terms | Terms in quotes ("occurrence", "insured") are resolved against the Definitions section and shown with the answer. |
| Exclusions | Always retrieved for coverage questions, since that's where most coverage misses happen. |

## Architecture

```
 Streamlit UI (local, or as a Databricks App)
        │
        ▼
 Coverage agent (LangGraph)
   1. classify question  (coverage / definition / limit / condition / out-of-scope)
   2. locate clauses     (structure-aware retrieval)
   3. resolve definitions
   4. draft answer       (open model via Groq)
   5. endorsement check
   6. citation audit     (unsupported claims → removed or refusal)
        │
        ▼
 Clause store ← structure parser ← policy forms (data/forms)
```

| Layer | Local | Azure Databricks |
|---|---|---|
| Clause store / retrieval | Chroma | Unity Catalog Delta table + Databricks Vector Search |
| LLM | Groq, key in `.env` | Groq through Databricks AI Gateway (guardrails, usage logging) |
| UI | `streamlit run` | Databricks App |
| Ingestion | Python CLI | Databricks job / notebook → Unity Catalog |
| Deploy | — | Databricks Asset Bundle via GitHub Actions (OIDC) |
| Evaluation | MLflow evaluation, gated in CI | same |

`CL_BACKEND=local|databricks` switches backends; the agent code is the same in both.

Design details: [docs/DESIGN.md](docs/DESIGN.md)

## Results

Evaluated on the 14-question set in `src/clauselens/eval/questions.jsonl`, run locally against
the sample forms (`python -m clauselens.eval.run`):

| Metric | Score | CI threshold |
|---|---|---|
| Verdict accuracy | 14/14 (1.00) | 0.75 |
| Citation recall (expected clause cited) | 1.00 | 0.70 |
| Out-of-scope refusals | 1/1 | 1.00 |
| Endorsement effect detected | 1/1 | 1.00 |

Plus 21 unit tests covering the parser, clause stores, citation audit and agent graph
(no API key needed). Model answers vary slightly between runs, so scores can move a little.

## Status

- [x] Policy structure parser (sections, outline clauses, defined terms, endorsement instructions)
- [x] Clause stores: in-memory BM25, Chroma, Databricks Vector Search
- [x] Coverage agent (LangGraph) with endorsement override check and citation audit
- [x] Streamlit UI
- [x] Evaluation set, scorers, MLflow run, CI gate
- [x] Databricks Asset Bundle: ingestion job, AI Gateway endpoint, Databricks App
- [ ] Deployed to an Azure Databricks workspace
- [ ] More forms (real specimen forms kept locally in `data/forms/private/`)

## Run it locally

Requires Python 3.11+ and a free [Groq](https://console.groq.com) API key.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[local,dev]"
cp .env.example .env                                   # then set GROQ_API_KEY

python -m clauselens.ingest --show                     # parse forms, index into Chroma
streamlit run src/clauselens/app/streamlit_app.py      # chat UI
python -m clauselens.eval.run                          # question set + thresholds
pytest && ruff check .                                 # tests (no API key needed)
```

Set `CL_LOCAL_STORE=memory` in `.env` to skip Chroma and embeddings entirely.

The Groq model is configurable with `CL_GROQ_MODEL` in `.env` (default
`llama-3.3-70b-versatile`; `openai/gpt-oss-120b` and `llama-3.1-8b-instant` also work).
The agent only depends on a small LLM interface (`agent/llm.py`), so other providers can be
added as another client class.

## Deploy to Azure Databricks

1. Create a secret scope `clauselens` with key `groq-api-key`.
2. Set your workspace URL in `databricks.yml`.
3. `databricks bundle deploy -t dev`, then `databricks bundle run ingest_forms -t dev`
   to build the clauses table and Vector Search index.
4. Grant the app's service principal `SELECT` on `clauselens_dev.policies.clauses` and
   `CAN_USE` on the Vector Search endpoint, then `databricks bundle run clauselens_app -t dev`.
5. For CI deploys, add `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` and
   `DATABRICKS_HOST` as GitHub environment secrets (OIDC federated credential) and run
   the "Deploy to Azure Databricks" workflow.

Vector Search endpoints and Apps bill while running; delete or stop them after testing.

## Repo layout

```
src/clauselens/
├── models.py        # PolicyForm, Clause, Citation, CoverageAnswer
├── parsing/         # structure-aware policy parser
├── retrieval/       # clause store interface + Chroma / Vector Search implementations
├── agent/           # LangGraph coverage agent
├── app/             # Streamlit UI (deployed as a Databricks App)
└── eval/            # coverage question set + MLflow scorers
notebooks/           # Databricks ingestion notebook
resources/           # bundle resources: ingest job, AI Gateway endpoint, app
databricks.yml, app.yaml, requirements.txt   # bundle root, app runtime config
.github/workflows/   # CI (tests, eval gate) and CD (bundle deploy)
data/forms/          # synthetic sample forms (see data/forms/SOURCES.md)
docs/DESIGN.md       # design decisions
```

## License

MIT (see LICENSE).
