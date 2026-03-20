# Databricks notebook source
# Pings the Whisper endpoint to prevent scale-to-zero during business hours

import base64
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Send a minimal request to keep the endpoint warm
try:
    response = w.serving_endpoints.query(
        name="whisper-transcriber",
        inputs=[{"audio": base64.b64encode(b"\x00").decode()}],
    )
    print("Endpoint is warm")
except Exception as e:
    # Even a failed request wakes the endpoint
    print(f"Ping sent (endpoint waking): {e}")
