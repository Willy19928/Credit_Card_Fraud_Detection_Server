#!/bin/sh
set -eu

for artifact in primary_mlp.pt preprocessing.joblib model_manifest.json; do
  if [ ! -f "/app/models/$artifact" ]; then
    cp "/app/default-models/$artifact" "/app/models/$artifact"
  fi
done

exec "$@"
