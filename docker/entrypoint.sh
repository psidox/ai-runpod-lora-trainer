#!/bin/bash
set -euo pipefail

TRAINING_BACKEND=${TRAINING_BACKEND:-ostris}
TRAINING_CONFIG_PATH=${TRAINING_CONFIG_PATH:-/workspace/dataset/training-config/config.toml}
SD_SCRIPTS_CONFIG=${SD_SCRIPTS_CONFIG:-/workspace/dataset/training-config/sd-config.json}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/output}
MODEL_PATH=${MODEL_PATH:-/workspace/models/model.safetensors}
TOOLKIT_PRESET=${TOOLKIT_PRESET:-z-image-turbo}
NETWORK_TYPE=${NETWORK_TYPE:-z-image-turbo}
KEEP_ALIVE=${KEEP_ALIVE:-0}
S3_BUCKET=${S3_BUCKET:-}
S3_OUTPUT_PREFIX=${S3_OUTPUT_PREFIX:-lora-outputs}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}

mkdir -p "$OUTPUT_DIR"

case "$TRAINING_BACKEND" in
  sd-scripts)
    echo "Running sd-scripts training using $SD_SCRIPTS_CONFIG"
    accelerate launch /workspace/sd-scripts/train_network.py --config "$SD_SCRIPTS_CONFIG"
    ;;
  ostris|*)
    echo "Running Ostris AI Toolkit training using $TRAINING_CONFIG_PATH"
    cd /workspace/ai-toolkit
    python -m aitoolkit.train_lora \
      --preset "$TOOLKIT_PRESET" \
      --dataset_config "$TRAINING_CONFIG_PATH" \
      --model "$MODEL_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --network_type "$NETWORK_TYPE"
    ;;
esac

if [ -n "$S3_BUCKET" ]; then
  echo "Uploading training artifacts to s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}"
  aws s3 sync "$OUTPUT_DIR" "s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}" --region "$AWS_DEFAULT_REGION"
fi

if [ "$KEEP_ALIVE" = "1" ]; then
  echo "Training complete. Keeping container alive for inspection."
  tail -f /dev/null
fi
