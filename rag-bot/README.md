# RAG бот

## Начальная настройка

Установить:
- [python >= 3.12](https://www.python.org/downloads/)
- [пакетный менеджер uv](https://docs.astral.sh/uv/getting-started/installation/)

Выполнить команды:
```bash
cd <project_dir>

# Подготовить виртуальное окружение для запуска
uv sync
source .venv/bin/activate

# Настроить переменные среды по образцу (rag-bot/.env.example)
cp rag-bot/.env.example rag-bot/.env
# Внести свой hugging-face токен в переменную, если без этого не работает
```

<details><summary>Про hugging-face токены</summary>
Можно довольно долго работать без токена (не указывать его в переменной среды HF_TOKEN).
Но если при запуске (в процессе создания модели эмбеддингов) возникает HTTP 429 (To many requests), то необходимо зарегистрироваться на https://huggingface.co/ и в разделе Settings->Tokens создать его и заполнить переменную HF_TOKEN в rag-bot/.env.
Для того, чтобы модель каждый раз не скачивалась из Интернета установлено HF_HUB_OFFLINE=1. В этом случае модель будут искать в локально кэше (~/.cache/huggingface/hub).
Но при первом запуске нужно выставить HF_HUB_OFFLINE=0, чтобы модель скачалась.
</details>

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

## Тестирование индекса векторной БД

Произвести [начальную настройку](#начальная-настройка).  

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Запустить тест
python3 -m rag_bot test_vdb <embeddings_model> <vector_db>
# python3 -m rag_bot test_embeddings intfloat/multilingual-e5-small chroma
```

Также доступны преднастроенные конфигурации запуска в [vscode](../.vscode/launch.json).  

## Тестирования моделей эмбеддингов

Произвести [начальную настройку](#начальная-настройка).

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Удалить ранее созданную БД (если есть)
rm -rf <project_dir>/vdb

# Запустить тест
python3 -m rag_bot test_embeddings <embeddings_model> <vector_db> <llm>
# python3 -m rag_bot test_embeddings intfloat/multilingual-e5-small chroma gemma4:e2b
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

## Тестирование бота

Произвести [начальную настройку](#начальная-настройка).

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Запустить тест
python3 -m rag_bot test_rag_bot <embeddings_model> <vector_db> <llm>
# python3 -m rag_bot test_rag_bot intfloat/multilingual-e5-small chroma gemma4:e2b
```

Также доступны преднастроенные конфигурации запуска в [vscode](../.vscode/launch.json).  
Список поддерживаемых LLM можно найти в [файле](./rag_bot/consts/llms.py).


## Запуск бота

Произвести [начальную настройку](#начальная-настройка).

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Запустить тест
python3 -m rag_bot run_bot <embeddings_model> <vector_db> <llm> <rag_mode>
# python3 -m rag_bot run_bot intfloat/multilingual-e5-small chroma gemma4:e2b minimal
```
rag_mode - режим RAG-пайплайна. Сейчас предусмотрены:
- minimal - отсутствуют техники промптинга;
- few_shot - добавление примеров (Few-Shot);
- cot - вывод рассуждений бота (Chain-of-Thought).

Также доступны преднастроенные конфигурации запуска в [vscode](../.vscode/launch.json).  

## Запуск бота c включенной защитой

Производится так же как обычный [запуск](#запуск-бота), лишь режим должен быть security (добавить проверки безопасности) или all (включить все опции сразу).
```bash
# проверки безопасности
python3 -m rag_bot run_bot intfloat/multilingual-e5-small chroma gemma4:e2b security

# все опции 
python3 -m rag_bot run_bot intfloat/multilingual-e5-small chroma gemma4:e2b all
```

## Запуск скрипта обновления индекса
Произвести [начальную настройку](#начальная-настройка).

Выполнить команды:
```bash
cd <project_dir>/rag-bot

# Запустить обновление
python3 -m rag_bot run_bot <embeddings_model> <vector_db>
# python -m rag_bot update_vdb intfloat/multilingual-e5-small chroma
```

## Добавление скрипта обновления индекса в cron

Для добавления/удаления записи в crontab можно воспользоваться скриптами из [папки](./rag-bot/scripts/).  
Необходимо скорректировать запись в crontab внутри [скрипта добавления](./scripts/add_vdb_updater_to_cron.sh).  

```bash
cd <project_dir>/rag-bot/scripts/

# добавление
./add_vdb_updater_to_cron.sh

# удаление
./remove_vdb_updater_from_cron.sh
```
