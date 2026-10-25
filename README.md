# ClauseLens

**Clause-cited coverage answers for insurance policies, built on Databricks.**

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
| Your own policy | Pick a policy and ask "when does my policy expire?" or "what will my renewal cost?". Answers come from the policy record, fetched through a policy API, and cite each field used. |

## Policy records

Besides the policy wording, ClauseLens can answer questions about a customer's own policy:
expiration date, renewal premium, next payment, status, deductible, which endorsements are
attached. Picking a policy in the sidebar also selects that policy's form and endorsements,
so coverage questions are answered against the right wording.

- Records come from a **policy system** behind a small interface (`policies/records.py`):
  a JSON file (default), an HTTP policy API, or a Unity Catalog table on Databricks.
- `python -m clauselens.policies.api` runs a sample policy API on port 8600; set
  `CL_POLICY_SOURCE=http` to have ClauseLens fetch records through it. In a company, point
  `CL_POLICY_API_URL` at the existing policy service instead.
- The model only restates values from the record (cited as `[P1]`, `[P2]` …); the citation
  audit removes any sentence that doesn't cite a field. It never calculates premiums.
- The three records in `data/policies/sample_policies.json` are synthetic sample data.
  Quoting new insurance is out of scope.

## Architecture

```
 Streamlit UI (local, or as a Databricks App)
        │
        ▼
 Coverage agent (LangGraph)
   1. classify question  (coverage / definition / limit / condition / account / out-of-scope)
      └─ account questions → policy record lookup (policy API) → cited answer
   2. locate clauses     (structure-aware retrieval)
   3. resolve definitions
   4. draft answer       (open model via Groq)
   5. endorsement check
   6. citation audit     (unsupported claims → removed or refusal)
        │
        ▼
 Clause store ← structure parser ← policy forms (data/forms)
```

| Layer | Local | Databricks |
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

The question set in `src/clauselens/eval/questions.jsonl` has 20 questions: 14 about the
policy wording and 6 about policy records (checked for the right dates and amounts). Run it
with `python -m clauselens.eval.run` or from the app's Test results page.

Results on the 14 policy-wording questions, run locally against the sample forms:

| Metric | Score | CI threshold |
|---|---|---|
| Verdict accuracy | 14/14 (1.00) | 0.75 |
| Citation recall (expected clause cited) | 1.00 | 0.70 |
| Out-of-scope refusals | 1/1 | 1.00 |
| Endorsement effect detected | 1/1 | 1.00 |

Plus 31 unit tests covering the parser, clause stores, citation audit, agent graph and
policy-record lookup (no API key needed). Model answers vary slightly between runs, so scores can move a little.

## Status

- [x] Policy structure parser (sections, outline clauses, defined terms, endorsement instructions)
- [x] Clause stores: in-memory BM25, Chroma, Databricks Vector Search
- [x] Coverage agent (LangGraph) with endorsement override check and citation audit
- [x] Streamlit UI
- [x] Policy-record questions (expiry, renewal, payments) through a policy API
- [x] Evaluation set, scorers, MLflow run, CI gate
- [x] Databricks Asset Bundle: ingestion job, AI Gateway endpoint, Databricks App
- [ ] Deployed to a Databricks workspace
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

## Deploy to Databricks

The bundle (`databricks.yml` + `resources/`) creates a Unity Catalog schema, a serving
endpoint with AI Gateway in front of Groq, an ingestion job, and the Databricks App.
Target `trial` is a Databricks-hosted workspace deployed from a laptop; target `azure` is an
Azure Databricks workspace deployed from GitHub Actions with OIDC.

1. Sign in: `databricks auth login --host <workspace URL>`.
2. Store the Groq key as a secret (never in code):
   `databricks secrets create-scope clauselens`, then
   `databricks secrets put-secret clauselens groq-api-key` and paste the key when prompted.
3. Set the SQL warehouse id (`warehouse_id`) in `databricks.yml`; the app reads
   `policy_records` through it.
4. `databricks bundle validate`, then `databricks bundle deploy`.
5. `databricks bundle run ingest_forms` builds the `clauses` and `policy_records` tables and
   the Vector Search index.
6. Give the app's service principal `USE CATALOG`, `USE SCHEMA` and `SELECT` on the schema,
   then `databricks bundle run clauselens_app`.

Vector Search endpoints and Apps bill while running; stop or delete them after testing
(`databricks bundle destroy` removes everything the bundle created).

## Repo layout

```
src/clauselens/
├── models.py        # PolicyForm, Clause, Citation, CoverageAnswer
├── parsing/         # structure-aware policy parser
├── retrieval/       # clause store interface + Chroma / Vector Search implementations
├── agent/           # LangGraph coverage agent
├── policies/        # policy records: file / HTTP API / Unity Catalog, plus a sample API
├── app/             # Streamlit UI (deployed as a Databricks App)
└── eval/            # coverage question set + MLflow scorers
notebooks/           # Databricks ingestion notebook
resources/           # bundle resources: ingest job, AI Gateway endpoint, app
databricks.yml, app.yaml, requirements.txt   # bundle root, app runtime config
.github/workflows/   # CI (tests, eval gate) and CD (bundle deploy)
data/forms/          # synthetic sample forms (see data/forms/SOURCES.md)
data/policies/       # synthetic sample policy records
docs/DESIGN.md       # design decisions
```

## License

MIT (see LICENSE).
