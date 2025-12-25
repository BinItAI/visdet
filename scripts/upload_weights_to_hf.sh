#!/bin/bash
# Upload visdet weights to HuggingFace Hub
#
# Usage:
#   1. Add your HuggingFace token below
#   2. Run: ./scripts/upload_weights_to_hf.sh

set -e

# ============================================
# CONFIGURATION - Edit these values
# ============================================

# Your HuggingFace token (get it from https://huggingface.co/settings/tokens)
HF_TOKEN="YOUR_TOKEN_HERE"

# The repository to upload to (format: org/repo-name)
REPO_ID="BinItAI/visdet-weights"

# ============================================
# Don't edit below this line
# ============================================

if [ "$HF_TOKEN" = "YOUR_TOKEN_HERE" ]; then
    echo "ERROR: Please edit this script and add your HuggingFace token"
    exit 1
fi

cd "$(dirname "$0")/.."

echo "Starting weight migration to HuggingFace..."
echo "Repository: $REPO_ID"
echo ""

python tools/migrate_weights_to_huggingface.py \
    --repo-id "$REPO_ID" \
    --token "$HF_TOKEN"

echo ""
echo "Done! Weights uploaded to https://huggingface.co/$REPO_ID"
