#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CWD="$PROJECT_DIR/rag-bot"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PY_MODULE="rag_bot"
EMBEDDINGS_MODEL="intfloat/multilingual-e5-small"
VDB_NAME="chroma"
CRON_TIME="17 * * * *"

# Полная команда
CMD="cd $CWD && $VENV_PYTHON -m $PY_MODULE update_vdb $EMBEDDINGS_MODEL $VDB_NAME"

# Временный файл для crontab
TEMP_CRON=$(mktemp)

# Сохраняем текущий crontab во временный файл
crontab -l > "$TEMP_CRON" 2>/dev/null

# Проверяем, не добавлена ли уже задача
if grep -q "update_vdb.*$EMBEDDINGS_MODEL" "$TEMP_CRON"; then
    echo "⚠️ Задача уже существует в crontab"
    rm "$TEMP_CRON"
    exit 1
fi

# Добавляем новую задачу
echo "$CRON_TIME $CMD" >> "$TEMP_CRON"

# Устанавливаем обновлённый crontab
crontab "$TEMP_CRON"

# Удаляем временный файл
rm "$TEMP_CRON"

echo "✅ Задача добавлена в crontab:"
echo "   $CRON_TIME $CMD"
