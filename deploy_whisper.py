# Databricks notebook source
# MAGIC %pip install mlflow torch torchaudio openai-whisper
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import numpy as np

# COMMAND ----------

# Define a custom MLflow PythonModel wrapper for Whisper
class WhisperTranscriber(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import whisper
        # Load from bundled artifacts — no network download on cold start
        model_dir = context.artifacts["whisper_model_dir"]
        self.model = whisper.load_model("base", download_root=model_dir) #change base to medium for larger model

    def predict(self, context, model_input):
        import base64
        import tempfile
        import os
        results = []
        for idx, row in model_input.iterrows():
            audio_b64 = row.get("audio", row.get("audio_base64", ""))
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            try:
                result = self.model.transcribe(tmp_path)
                results.append(result["text"])
            finally:
                os.unlink(tmp_path)
        return results

# COMMAND ----------

import whisper
import os

# Define model signature
input_schema = Schema([ColSpec(DataType.string, "audio")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Download model weights locally (one-time); bundled as artifact to avoid
# re-downloading on every serving endpoint cold start
model_dir = "/tmp/whisper_weights"
os.makedirs(model_dir, exist_ok=True)
whisper.load_model("base", download_root=model_dir) #change base to medium for larger model

# Log and register the model
catalog = "uplight_demo_gen_catalog"
schema = "watlow_ingestion"
model_name = f"{catalog}.{schema}.whisper_transcriber"

with mlflow.start_run(run_name="whisper_base_deploy"):
    mlflow.pyfunc.log_model(
        artifact_path="whisper_model",
        python_model=WhisperTranscriber(),
        registered_model_name=model_name,
        signature=signature,
        artifacts={"whisper_model_dir": model_dir},
        pip_requirements=[
            "openai-whisper",
            "torch>=2.0,<3",
            "torchaudio>=2.0,<3",
            "numpy<2",
        ],
    )

print(f"Model registered as: {model_name}")

# COMMAND ----------

# Get latest model version using search_model_versions (UC compatible)
from mlflow import MlflowClient
from datetime import timedelta
client = MlflowClient()
versions = client.search_model_versions(f"name='{model_name}'")
model_version = max(v.version for v in versions)
print(f"Deploying model version: {model_version}")

# Create serving endpoint via Databricks SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

served_entity = ServedEntityInput(
    entity_name=model_name,
    entity_version=str(model_version),
    workload_size="Small", #increase this for faster processing (transcription accuracy is good, just slow)
    workload_type="GPU_SMALL", #increase this and switch to medium whisper model for better transcription accuracy
    scale_to_zero_enabled=True,
)

try:
    w.serving_endpoints.create_and_wait(
        name="whisper-transcriber",
        config=EndpointCoreConfigInput(served_entities=[served_entity]),
        timeout=timedelta(minutes=40),
    )
    print("Endpoint 'whisper-transcriber' created and ready!")
except Exception as e:
    if "already exists" in str(e).lower():
        print("Endpoint already exists, updating...")
        w.serving_endpoints.update_config_and_wait(
            name="whisper-transcriber",
            served_entities=[served_entity],
            timeout=timedelta(minutes=40),
        )
        print("Endpoint updated!")
    else:
        raise e