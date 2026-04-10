from dataclasses import dataclass
import hashlib
import os
import sys
from pathlib import Path
from logging import basicConfig, INFO, getLogger, StreamHandler
from logging.handlers import RotatingFileHandler
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

import rag_bot.settings as st
from rag_bot.rag_creator import split_documents
from rag_bot.logs import APP_LOG_LEVEL

LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_NAME = 'index_updater.log'
FULL_LOG_PATH = Path(LOG_DIR) / LOG_FILE_NAME

UPDATE_BATCH_SIZE = 5000

basicConfig(
    level=APP_LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        StreamHandler(sys.stdout),
        RotatingFileHandler(FULL_LOG_PATH, maxBytes=1_000_000, encoding='utf-8')
    ],
    force=True
)
logger = getLogger(__name__)


def update_index(vector_db: Chroma, kdb_path: str, embeddings_data: st.EmbeddingsModelData):
    updater = ChromaFileSync(vector_db, kdb_path)
    updater.sync(embeddings_data.chunk_size, embeddings_data.chunk_overlap)

SOURCELESS_CHUNK = f'Chunk without source found'

@dataclass(frozen=True, kw_only=True, eq=False)
class UpdaterStat:
    added_count: int
    updated_count: int
    chunks_count: int
    deleted_count: int


class ChromaFileSync:
    def __init__(self, vector_db: Chroma, kdb_path: str):
        self._vector_db = vector_db
        self._kdb_path = Path(kdb_path)

    def _get_all_file_hashes_from_db(self) -> dict[str, dict]:
        result = self._vector_db.get(include=['metadatas'])

        file_info = {}
        for meta, chunk_id in zip(result['metadatas'], result['ids']):
            file_path = meta.get('source')
            if not file_path:
                logger.warning(f'{SOURCELESS_CHUNK}.({chunk_id})')
                continue

            if file_path not in file_info:
                file_info[file_path] = {
                    'file_hash': meta.get('file_hash'),
                    'chunk_ids': []
                }
            file_info[file_path]['chunk_ids'].append(chunk_id)

        return file_info

    def _get_current_files_on_disk(self) -> dict[str, str]:
        current_files = {}
        for file_path in self._kdb_path.rglob('*.txt'):
            file_path_str = str(file_path)

            with open(file_path_str, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            current_files[file_path_str] = file_hash

        return current_files

    def _group_files(
        self, disk_files: dict[str, str], db_files: dict[str, dict],
    ) -> tuple[Iterable[str]]:

        files_to_add = []
        files_to_update = []
        files_to_delete = []

        for file_path in disk_files:
            if file_path not in db_files:
                files_to_add.append(file_path)
            elif disk_files[file_path] != db_files[file_path]['file_hash']:
                files_to_update.append(file_path)

        for file_path in db_files:
            if file_path not in disk_files:
                files_to_delete.append(file_path)

        return files_to_add, files_to_update, files_to_delete

    def _delete_old_chunks(
        self,
        db_files: dict[str, dict],
        files_to_delete: Iterable[str],
        files_to_update: Iterable[str],
    ):
        for file_path in files_to_delete:
            chunk_ids = db_files[file_path]['chunk_ids']
            self._vector_db.delete(ids=chunk_ids)
            logger.app_info(f'Удалён файл: {file_path} (фрагментов: {len(chunk_ids)})')

        for file_path in files_to_update:
            chunk_ids = db_files[file_path]['chunk_ids']
            self._vector_db.delete(ids=chunk_ids)
            logger.app_info(f'Удалены старые фрагменты изменённого файла: {file_path}')

    def _load_documents(self, files_to_load: Iterable[str]) -> Iterable[Document]:
        documents = []

        for file_path in files_to_load:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)

        return documents
    
    def _sync(self, chunk_size: int, chunk_overlap: int) -> UpdaterStat:
        disk_files = self._get_current_files_on_disk()
        db_files = self._get_all_file_hashes_from_db()

        files_to_add, files_to_update, files_to_delete = self._group_files(disk_files, db_files)
        self._delete_old_chunks(db_files, files_to_delete, files_to_update)
        files_to_load = files_to_add + files_to_update

        documents = self._load_documents(files_to_load)
        if chunks := split_documents(documents, chunk_size, chunk_overlap):
            for i in range(0, len(chunks), UPDATE_BATCH_SIZE):
                batch = chunks[i:i + UPDATE_BATCH_SIZE]
                self._vector_db.add_documents(documents=batch)

        return UpdaterStat(
            added_count=len(files_to_add),
            updated_count=len(files_to_update),
            chunks_count=len(chunks),
            deleted_count=len(files_to_delete),
        )

    def _failure_notification(ex: Exception):
        """Уведмление о неудачном обновлении."""
        ...  # Предполагается реализация в зависимость от доступной инфраструктуры (SMS, EMail, Broker Msg и другие)

    def sync(self, chunk_size: int, chunk_overlap: int):
        logger.app_info('Запущен процесс обновления векторного индекса')

        try:
            updater_stat = self._sync(chunk_size, chunk_overlap)
        except Exception as ex:
            logger.error(ex)
            self._failure_notification(ex)
        else:
            logger.app_info(f'Добавлено файлов: {updater_stat.added_count}')
            logger.app_info(f'Обновлено файлов: {updater_stat.updated_count}')
            logger.app_info(f'Добавлено/обновлено чанков: {updater_stat.chunks_count}')
            logger.app_info(f'Удалено файлов: {updater_stat.deleted_count}')

        logger.app_info(f'Обновление векторного индекса завершено')
