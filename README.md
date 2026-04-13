# Multimodal Transcription App

A Databricks App that ingests documents (PDFs, images, audio, video) and extracts their content using AI. Built with FastAPI + React.

## What It Does

- Upload PDFs, images (PNG/JPG/TIFF/BMP), and MP3 audio files through a web portal
- PDFs and images are parsed via `ai_parse_document` on a SQL warehouse
- Audio files are transcribed via the Whisper Large V3 model on a serving endpoint
- All files are stored in a Unity Catalog Volume with parsed content in a Delta table

## Architecture

```
React Frontend
  → FastAPI Backend (Databricks App)
      → UC Volume (file storage)
      → SQL Warehouse (ai_parse_document for PDFs/images)
      → Model Serving Endpoint (Whisper V3 for audio transcription)
      → Delta Table (parsed content + metadata)
```

## Project Structure

```
├── app.yaml                 # Databricks App config (env vars, resources)
├── requirements.txt         # Python dependencies
├── backend/
│   └── main.py              # FastAPI app — upload, parse, transcribe, CRUD
├── frontend/
│   └── build/               # React frontend (pre-built static files)
├── databricks.yml            # DAB config — bundle variables and deployment targets
├── resources/
│   ├── app.yml               # App resource definition (env vars, SQL warehouse)
│   └── jobs.yml              # Job resources (deploy whisper, keepalive schedule)
├── deploy_whisper.py         # Notebook: deploy Whisper V3 from system.ai
├── batch_transcribe.py       # Notebook: batch transcribe unprocessed MP3s offline
├── keepalive.py              # Notebook: ping whisper endpoint to prevent scale-to-zero
└── CLAUDE.md                 # Project context for Claude Code AI assistant
```

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- SQL Warehouse
- `system.ai.whisper_large_v3` model available in Unity Catalog — install via the [Databricks Marketplace](https://marketplace.databricks.com/details/1eceaa77-6b60-42f0-9809-ceccf1b237f5/Databricks_Whisper-V3-Model)

## Setup

### 1. Deploy the Whisper endpoint

Run `deploy_whisper.py` as a notebook or job. This creates a `whisper-transcriber` model serving endpoint using the Whisper Large V3 model from `system.ai`.

### 2. Configure app.yaml

Update the environment variables in `app.yaml`:

| Variable | Description |
|----------|-------------|
| `CATALOG` | Unity Catalog catalog name |
| `SCHEMA` | Schema for tables and volumes |
| `VOLUME` | Volume name for raw document storage |
| `PARSED_TABLE` | Delta table name for parsed content |
| `DATABRICKS_WAREHOUSE_ID` | SQL Warehouse ID (via `sql-warehouse` resource) |

### 3. Deploy via Databricks Asset Bundle

The project includes a `databricks.yml` for reproducible multi-workspace deployment. It deploys the app, a Whisper deploy job, and a keepalive scheduled job.

**Variables** (override per target in `databricks.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `catalog` | `uplight_demo_gen_catalog` | Unity Catalog catalog |
| `schema` | `watlow_ingestion` | Schema for tables and volumes |
| `volume` | `raw_documents` | Volume for raw document storage |
| `parsed_table` | `parsed_documents_gemini` | Delta table for parsed content |
| `whisper_endpoint` | `whisper-transcriber` | Model serving endpoint name |

**Deploy:**

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

**Targets:** `dev` (default), `prod` — both configurable in `databricks.yml`.

**What gets deployed:**
- The Databricks App (with parameterized env vars)
- `[dev] Deploy Whisper Endpoint` job — one-time notebook to create/update the Whisper serving endpoint
- `[dev] Whisper Keepalive` job — scheduled every 30 min, Mon–Fri 8am–5pm CT

Alternatively, deploy the app manually:

```bash
databricks apps create --name watlow-knowledge-ingestion
databricks apps deploy watlow-knowledge-ingestion --source-code-path /Workspace/path/to/this/project
```

### 4. (Optional) Batch transcribe

Run `batch_transcribe.py` as a notebook or scheduled job to bulk-transcribe MP3 files that haven't been processed yet. This is useful for:

- Backfilling historical audio files already in the volume
- Reprocessing files that failed during real-time upload
- Bulk ingestion of large audio datasets

The notebook scans the UC Volume for unprocessed MP3s, transcribes them via `ai_query` on the SQL warehouse, and merges results into the parsed documents table. Individual file failures are handled gracefully — the batch continues and failed files are marked with error details.

Override defaults via Databricks widgets or job parameters: `catalog`, `schema`, `volume`, `parsed_table`, `whisper_endpoint`.

### 5. (Optional) Schedule keepalive

Run `keepalive.py` on a schedule (e.g., every 30 minutes during business hours) to keep the Whisper endpoint warm and avoid cold-start latency.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload and parse a file |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}` | Get document details + parsed content |
| `DELETE` | `/api/documents/{id}` | Delete a document |
