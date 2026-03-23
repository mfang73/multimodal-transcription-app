# Databricks notebook source
# Deploys the Whisper Large V3 model from the system.ai catalog
# as a model serving endpoint for audio transcription.

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import datetime
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

# Model is pre-registered in the system.ai catalog
model_uc_path = "system.ai.whisper_large_v3"

w = WorkspaceClient()

# Automatically retrieve the latest model version
versions = list(w.model_versions.list(full_name=model_uc_path))
latest_version = str(max(int(v.version) for v in versions))
print(f"Deploying {model_uc_path} version {latest_version}")

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": "whisper-transcriber",
            "model_name": model_uc_path,
            "model_version": latest_version,
            "workload_type": "GPU_MEDIUM",
            "workload_size": "Small",
            "scale_to_zero_enabled": "True",
        }
    ]
})

def _update_endpoint(retries=5, wait_seconds=30):
    for attempt in range(retries):
        try:
            w.serving_endpoints.update_config_and_wait(
                name="whisper-transcriber",
                served_entities=[
                    ServedEntityInput(
                        name="whisper-transcriber",
                        entity_name=model_uc_path,
                        entity_version=latest_version,
                        workload_type=ServingModelWorkloadType.GPU_MEDIUM,
                        workload_size="Small",
                        scale_to_zero_enabled=True,
                    )
                ],
                timeout=datetime.timedelta(minutes=90),
            )
            print("Endpoint updated!")
            return
        except Exception as ex:
            if "currently being updated" in str(ex).lower() and attempt < retries - 1:
                print(f"Endpoint busy, retrying in {wait_seconds}s ({attempt + 1}/{retries})...")
                time.sleep(wait_seconds)
            else:
                raise

try:
    model_details = w.serving_endpoints.create(name="whisper-transcriber", config=config)
    model_details.result(timeout=datetime.timedelta(minutes=90))
    print("Endpoint 'whisper-transcriber' created and ready!")
except Exception as e:
    if "already exists" in str(e).lower():
        print("Endpoint already exists, updating...")
        _update_endpoint()
    else:
        raise e