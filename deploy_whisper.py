# Databricks notebook source
# Deploys the Whisper Large V3 model from the system.ai catalog
# as a model serving endpoint for audio transcription.

# COMMAND ----------

%pip install --upgrade databricks-sdk
dbutils.library.restartPython()

# COMMAND ----------

import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput

# Model is pre-registered in the system.ai catalog
model_uc_path = "system.ai.whisper_large_v3"
version = "1"

w = WorkspaceClient()

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": "whisper-transcriber",
            "model_name": model_uc_path,
            "model_version": version,
            "workload_type": "GPU_MEDIUM",
            "workload_size": "Small",
            "scale_to_zero_enabled": "True",
        }
    ]
})

try:
    model_details = w.serving_endpoints.create(name="whisper-transcriber", config=config)
    model_details.result(timeout=datetime.timedelta(minutes=40))
    print("Endpoint 'whisper-transcriber' created and ready!")
except Exception as e:
    if "already exists" in str(e).lower():
        print("Endpoint already exists, updating...")
        w.serving_endpoints.update_config_and_wait(
            name="whisper-transcriber",
            served_entities=[{
                "name": "whisper-transcriber",
                "entity_name": model_uc_path,
                "entity_version": version,
                "workload_type": "GPU_MEDIUM",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }],
            timeout=datetime.timedelta(minutes=40),
        )
        print("Endpoint updated!")
    else:
        raise e
