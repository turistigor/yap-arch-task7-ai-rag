#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CWD="$PROJECT_DIR/rag-bot"
EMBEDDINGS_MODEL="intfloat/multilingual-e5-small"

TEMP_CRON=$(mktemp)
crontab -l > "$TEMP_CRON" 2>/dev/null

grep -v "update_vdb.*$EMBEDDINGS_MODEL" "$TEMP_CRON" > "$TEMP_CRON.tmp"
mv "$TEMP_CRON.tmp" "$TEMP_CRON"

crontab "$TEMP_CRON"
rm "$TEMP_CRON"

echo "✅ Задача удалена из crontab"