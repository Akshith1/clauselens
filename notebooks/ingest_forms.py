# Databricks notebook source
# MAGIC %md
# MAGIC # ClauseLens: ingest policy forms into Unity Catalog
# MAGIC
# MAGIC Parses every form in `data/forms/catalog.json` with the same parser used locally,
# MAGIC writes one row per clause to `<catalog>.<schema>.clauses`, and creates (or syncs)
# MAGIC the Vector Search index the app queries. Embeddings are computed by Databricks from
# MAGIC the `search_text` column.

# COMMAND ----------

# MAGIC %pip install pypdf databricks-vectorsearch --quiet

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "clauselens")
dbutils.widgets.text("vs_endpoint", "clauselens-vs")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
vs_endpoint = dbutils.widgets.get("vs_endpoint")
embedding_endpoint = dbutils.widgets.get("embedding_endpoint")
table = f"{catalog}.{schema}.clauses"
index_name = f"{catalog}.{schema}.clauses_index"

# COMMAND ----------

import os
import sys
from pathlib import Path

# The bundle deploys the whole repo; make the package importable from this notebook.
repo_root = Path(os.getcwd()).parent
sys.path.insert(0, str(repo_root / "src"))

from clauselens.parsing.loader import load_forms
from clauselens.retrieval.base import form_to_records

forms = load_forms(repo_root / "data" / "forms")
rows = [record.to_row() for form in forms for record in form_to_records(form)]
print(f"Parsed {len(forms)} forms into {len(rows)} clauses")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
(spark.createDataFrame(rows)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(table))
spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
display(spark.table(table).groupBy("form_number", "section_type").count())

# COMMAND ----------

# MAGIC %md
# MAGIC Sample customer policy records (synthetic) → `<catalog>.<schema>.policy_records`.
# MAGIC The app reads this table through a SQL warehouse to answer questions like
# MAGIC "when does my policy expire?". In a real company this table would be fed from the
# MAGIC policy administration system instead.

# COMMAND ----------

import json

from pyspark.sql import types as T

policy_fields = [
    ("policy_number", T.StringType()), ("insured_name", T.StringType()),
    ("form", T.StringType()), ("endorsements", T.StringType()), ("status", T.StringType()),
    ("effective_date", T.StringType()), ("expiration_date", T.StringType()),
    ("annual_premium", T.DoubleType()), ("renewal_premium", T.DoubleType()),
    ("renewal_offer_date", T.StringType()), ("deductible", T.DoubleType()),
    ("coverage_a_limit", T.DoubleType()), ("payment_plan", T.StringType()),
    ("next_payment_due", T.StringType()), ("next_payment_amount", T.DoubleType()),
]
policy_schema = T.StructType([T.StructField(n, t) for n, t in policy_fields])
policies = json.loads((repo_root / "data" / "policies" / "sample_policies.json").read_text())
policy_rows = [
    tuple(json.dumps(p.get(n) or []) if n == "endorsements"
          else (float(p[n]) if t == T.DoubleType() and p.get(n) is not None else p.get(n))
          for n, t in policy_fields)
    for p in policies]
(spark.createDataFrame(policy_rows, policy_schema)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.policy_records"))
display(spark.table(f"{catalog}.{schema}.policy_records"))

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

if vs_endpoint not in [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]:
    vsc.create_endpoint_and_wait(name=vs_endpoint, endpoint_type="STANDARD")

existing = [i["name"] for i in vsc.list_indexes(vs_endpoint).get("vector_indexes", [])]
if index_name in existing:
    vsc.get_index(vs_endpoint, index_name).sync()
    print(f"Triggered sync of {index_name}")
else:
    vsc.create_delta_sync_index_and_wait(
        endpoint_name=vs_endpoint,
        index_name=index_name,
        source_table_name=table,
        primary_key="record_id",
        pipeline_type="TRIGGERED",
        embedding_source_column="search_text",
        embedding_model_endpoint_name=embedding_endpoint,
    )
    print(f"Created {index_name}")
