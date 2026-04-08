# RAG бот

## Начальная настройка

Установить:
- [python >= 3.12](https://www.python.org/downloads/)
- [пакетный менеджер uv](https://docs.astral.sh/uv/getting-started/installation/)

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Подготовить виртуальное окружение для запуска
uv sync
source .venv/bin/activate
```

## Создание индекса векторной БД

Произвести [начальную настройку](#начальная-настройка).  

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Удалить ранее созданную БД (если есть)
rm -rf <project_dir>/vdb

# Запустить тест
python3 -m rag_bot create_vdb <embeddings_model> <vector_db>
# python3 -m rag_bot test_embeddings intfloat/multilingual-e5-small chroma
```

Также доступны преднастроенные конфигурации запуска в [vscode](../.vscode/launch.json).  

В результате выполнения в корне проекта появится папка vdb с векторным индексом.

## Тестирвоание индекса векторной БД

Произвести [начальную настройку](#начальная-настройка).  

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Запустить тест
python3 -m rag_bot test_vdb <embeddings_model> <vector_db>
# python3 -m rag_bot test_embeddings intfloat/multilingual-e5-small chroma
```

Также доступны преднастроенные конфигурации запуска в [vscode](../.vscode/launch.json).  

## Тестирования моделей эмбеддингнов

Произвести [начальную настройку](#начальная-настройка).

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Удалить ранее созданную БД (если есть)
rm -rf <project_dir>/vdb

# Запустить тест
python3 -m rag_bot test_embeddings <embeddings_model> <vector_db> <llm>
# python3 -m rag_bot test_embeddings intfloat/multilingual-e5-small chroma qwen2.5:3b
```

Также доступны преднастроенные конфигурации запуска в [vscode](../.vscode/launch.json).  
Список поддерживаемых моделей эмбеддингов можно найти в [файле](./rag_bot/consts/embeddings.py).

Анализ результатов производится на основе консольных логов.

1. Время создания БД в минутах:
```bash
VDB creation time, min: 0.10293240000000001
```

2. Процент успешного поиска документов вопросов:
```bash
2026-04-07 12:28:44,053 - INFO - Simple questions:
...
2026-04-07 12:28:44,321 - INFO - Average found percent: 90.00
2026-04-07 12:28:44,321 - INFO - Complex questions:
...
2026-04-07 12:28:44,431 - INFO - Average found percent: 70.00
```

## Тестированияе LLM

Произвести [начальную настройку](#начальная-настройка).

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Подготовить виртуальное окружение для запуска
uv sync
source .venv/bin/activate

# Запустить тест
python3 -m rag_bot test_rag_bot <embeddings_model> <vector_db> <llm>
# python3 -m rag_bot test_rag_bot intfloat/multilingual-e5-small chroma qwen2.5:3b
```

Также доступны преднастроенные конфигурации запуска в [vscode](../.vscode/launch.json).  
Список поддерживаемых LLM можно найти в [файле](./rag_bot/consts/llms.py).

Анализ результатов производится на основе консольных логов:
1. Среднее время выполнения было рассчитано вручную.
2. Качество ответов было проанализировано вручную.
