# ClauseLens — Design

## 1. Problem

Coverage questions are answered today by reading long policy forms plus endorsements
that silently modify them. Errors usually come from three places:

1. Missing an exclusion or condition far from the insuring agreement.
2. Not applying a defined term's narrow meaning.
3. Not noticing that an endorsement changes the base form.

ClauseLens targets those three failure modes explicitly instead of treating the policy
as generic text.

## 2. Goals and non-goals

**Goals**
- Answer coverage, definition, and limit questions with clause-level citations.
- Detect endorsement overrides and say how they change the answer.
- Refuse when the documents don't support an answer.
- Run the same agent locally and on Databricks (Azure or Databricks-hosted) with only a config change.

**Non-goals**
- Claims adjudication or binding coverage decisions. Answers are informational.
- Rating, quoting, or policy issuance.

## 3. Data

Public sample forms and endorsements from state insurance department filings and
other publicly published specimen policies. Every file in `data/forms/` is listed in
`data/forms/SOURCES.md` with its source URL and retrieval date. No private or
proprietary data is used.

## 4. Domain model (`models.py`)

- **PolicyForm**: form number, edition date, line of business, title, sections.
- **Section**: type (`declarations`, `insuring_agreement`, `definitions`,
  `exclusions`, `conditions`, `endorsement`), heading, clauses.
- **Clause**: hierarchical id (e.g. `I.A.3.b`), text, defined terms used, parent path.
- **Endorsement**: a PolicyForm plus `modifies` (target form numbers) and
  per-clause actions: `adds`, `deletes`, `replaces`, `limits`.
- **Citation**: form number, section, clause id, quoted span offsets.
- **CoverageAnswer**: verdict (`covered`, `not_covered`, `depends`, `cannot_determine`),
  explanation, citations, definitions used, endorsement effects.

## 5. Parsing

Policy forms follow strong conventions: numbered/lettered outlines, ALL-CAPS section
headings, quoted defined terms. The parser:

1. Normalizes text (PDF → text, page headers/footers removed).
2. Detects section boundaries from heading patterns.
3. Builds the clause tree from outline markers (`A.`, `1.`, `a.`, `(1)`).
4. Extracts defined terms from the Definitions section and tags their uses.
5. For endorsements, reads the "This endorsement modifies…" block and the
   replace/delete/add instructions.

Each clause is indexed with its full parent path so retrieval never returns an
orphaned sub-clause.

## 6. Retrieval

Index: one row per clause in a Delta table, served by Databricks Vector Search
(Chroma locally).
Filterable fields: form number, section type, line of business.

Retrieval strategy per question type:
- **Coverage**: insuring agreement and exclusions, plus conditions for the matched peril.
- **Definition**: definitions section first.
- **Limit**: declarations and limits-of-insurance sections.

Exclusions are always retrieved for coverage questions, since that's where most
misses happen.

## 7. Agent steps

1. **Classify** the question type, or route out-of-scope questions to a refusal.
2. **Locate** clauses using the type-specific strategy above.
3. **Resolve definitions** for defined terms in the retrieved clauses.
4. **Draft** the answer with Llama (via Groq) using only retrieved clauses, with citations.
5. **Endorsement check**: for each attached endorsement that modifies a cited clause,
   re-evaluate and record the effect.
6. **Citation audit**: each sentence must map to a cited clause span; unsupported
   sentences are removed, and if nothing remains the verdict is `cannot_determine`.

**Account questions** ("when does my policy expire?", "what will my renewal cost?") skip
retrieval. The agent fetches the selected policy's record from the policy system, numbers
its fields `[P1]…[Pn]`, asks the model to answer using only those fields, and runs the same
citation audit. With no policy selected it asks the user to pick one; it never guesses a
date or amount. Selecting a policy also sets the form and endorsements used for coverage
questions.

The policy system is an interface with three implementations: a JSON file, an HTTP client
for a policy API (a stdlib mock is included in `policies/api.py`), and a Unity Catalog
table read through a SQL warehouse. All records are synthetic. A real deployment would
point the HTTP client at an existing policy service; quoting new insurance is out of scope.

## 8. Platform and deployment

| Concern | Local development | Databricks |
|---|---|---|
| Clause store | Chroma, on disk | Unity Catalog Delta table `workspace.clauselens.clauses` |
| Retrieval | Chroma similarity + metadata filters | Databricks Vector Search (Delta Sync index) |
| LLM | Llama via Groq, key from `.env` | Llama via Groq, called through Databricks AI Gateway (guardrails, usage logging, key in a secret scope) |
| Agent | LangGraph | LangGraph (same code) |
| UI | Streamlit, run locally | Streamlit as a Databricks App |
| Policy records | JSON file or sample HTTP policy API | Unity Catalog table `policy_records` via SQL warehouse |
| Ingestion | Python CLI | Databricks job / notebook writing to Unity Catalog |
| Deploy | — | Databricks Asset Bundle, deployed from GitHub Actions with OIDC (no stored cloud secrets) |

A single setting (`CL_BACKEND=local|databricks`) selects the clause store and LLM
client; the agent code doesn't change between environments.

## 9. Evaluation

A hand-written question set over the public forms, each with expected verdict and
expected clause ids. Run with MLflow evaluation: custom scorers for verdict accuracy,
citation precision/recall, endorsement effect detection, refusal correctness, and
correct dates and amounts for policy-record questions, plus
built-in groundedness and relevance judges. GitHub Actions fails the build below agreed
thresholds.

## 10. Open decisions

- PDF extraction library choice (layout fidelity vs. dependencies).
- Embedding model and whether to embed clauses with their parent path text.
- How strict the citation audit should be (sentence-level vs. claim-level).
