# Databricks notebook source
# Batch transcribe MP3 files in the UC Volume that haven't been processed yet.
# Schedule this as a job to catch bulk uploads or reprocess failed transcriptions.

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Configuration — override via job parameters or widgets
_DEFAULTS = {
    "catalog": "uplight_demo_gen_catalog",
    "schema": "watlow_ingestion",
    "volume": "raw_documents",
    "parsed_table": "parsed_documents_gemini",
    "whisper_endpoint": "whisper-transcriber",
}
_widget_names = {w.name for w in dbutils.widgets.getAll()}

def _get_param(name):
    return dbutils.widgets.get(name) if name in _widget_names else _DEFAULTS[name]

CATALOG = _get_param("catalog")
SCHEMA = _get_param("schema")
VOLUME = _get_param("volume")
PARSED_TABLE = _get_param("parsed_table")
WHISPER_ENDPOINT = _get_param("whisper_endpoint")

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.{PARSED_TABLE}"

print(f"Volume: {VOLUME_PATH}")
print(f"Table: {TABLE_NAME}")
print(f"Endpoint: {WHISPER_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find unprocessed MP3 files
# MAGIC Compare files in the volume against records in the parsed table to find
# MAGIC MP3s that are new, failed, or still stuck in processing.

# COMMAND ----------

# Get all MP3 files in the volume
from pyspark.sql import Row
files = dbutils.fs.ls(VOLUME_PATH)
mp3_rows = [Row(volume_path=f.path) for f in files if f.path.lower().endswith('.mp3')]
all_files_df = spark.createDataFrame(mp3_rows) if mp3_rows else spark.createDataFrame([], "volume_path: string")

# Get already-completed MP3s from the table
completed_df = spark.sql(f"""
  SELECT volume_path
  FROM {TABLE_NAME}
  WHERE file_type = '.mp3'
    AND parse_status = 'completed'
    AND parsed_content IS NOT NULL
    AND parsed_content != ''
""")

# Find MP3s that need transcription (new, failed, or stuck)
unprocessed_df = all_files_df.join(completed_df, on="volume_path", how="left_anti")
unprocessed_count = unprocessed_df.count()
print(f"Found {unprocessed_count} MP3 files to transcribe")

if unprocessed_count == 0:
    dbutils.notebook.exit("No files to process")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch transcribe using ai_query

# COMMAND ----------

# Transcribe unprocessed MP3s and prepare upsert records in a single pass.
# Combining transcription + field extraction avoids re-running expensive ai_query calls.
# failOnError => FALSE lets the batch continue even if individual files fail.
unprocessed_df.createOrReplaceTempView("unprocessed_mp3s")

results_df = spark.sql(f"""
  WITH raw_results AS (
    SELECT
      u.volume_path,
      ai_query(
        '{WHISPER_ENDPOINT}',
        f.content,
        returnType => 'STRING',
        failOnError => FALSE
      ) AS transcription
    FROM unprocessed_mp3s u
    JOIN read_files('{VOLUME_PATH}/*.mp3', format => 'binaryFile') f
      ON f.path = u.volume_path
  )
  SELECT
    regexp_extract(volume_path, '/([^/]+)\\.mp3$', 1) AS document_id,
    regexp_extract(volume_path, '/([^/]+\\.mp3)$', 1) AS filename,
    '.mp3' AS file_type,
    current_timestamp() AS upload_timestamp,
    volume_path,
    transcription.result AS parsed_content,
    CASE
      WHEN transcription.errorMessage IS NULL THEN 'completed'
      ELSE 'failed'
    END AS parse_status,
    COALESCE(transcription.errorMessage, 'batch_transcribe') AS parse_metadata
  FROM raw_results
""")

results_df.createOrReplaceTempView("batch_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert results into the parsed table

# COMMAND ----------

# Merge — update existing failed/processing records, insert new ones
merge_result = spark.sql(f"""
  MERGE INTO {TABLE_NAME} t
  USING batch_results s
  ON t.volume_path = s.volume_path
  WHEN MATCHED AND t.parse_status != 'completed' THEN
    UPDATE SET
      t.parsed_content = s.parsed_content,
      t.parse_status = s.parse_status,
      t.parse_metadata = s.parse_metadata
  WHEN NOT MATCHED THEN
    INSERT (document_id, filename, file_type, upload_timestamp, volume_path, parsed_content, parse_status, parse_metadata)
    VALUES (s.document_id, s.filename, s.file_type, s.upload_timestamp, s.volume_path, s.parsed_content, s.parse_status, s.parse_metadata)
""")

display(merge_result)
print("Batch transcription complete.")