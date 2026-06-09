#!/bin/sh
set -eu

for artifact in primary_mlp.pt preprocessing.joblib model_manifest.json run_metadata.json; do
  if [ ! -f "/app/models/$artifact" ]; then
    if [ ! -w "/app/models" ]; then
      echo "Missing /app/models/$artifact and /app/models is not writable." >&2
      exit 1
    fi
    cp "/app/default-models/$artifact" "/app/models/$artifact"
  fi
done

exec "$@"
