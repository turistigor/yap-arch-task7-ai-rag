import json
import logging
import os
import sys
from logging import Handler, getLogger, StreamHandler, Formatter, LogRecord, FileHandler
from logging.handlers import RotatingFileHandler
from typing import Optional

APP_LOG_LEVEL = logging.INFO + 1

logging.addLevelName(APP_LOG_LEVEL, 'APP_INFO')

def app_info(self, msg, *args, **kwargs):
     if self.isEnabledFor(APP_LOG_LEVEL):
        self._log(APP_LOG_LEVEL, msg, args, **kwargs)

logging.Logger.app_info = app_info


class JSONLFormatter(logging.Formatter):
    def format(self, record: LogRecord):
        log_record = {
            'time': self.formatTime(record),
            'question': record.question,
            'answer_is_good': record.answer_is_good,
            'semantic similarity': record.semantic_similarity,
            'length difference': record.length_difference,
            'retrieval rate': record.retrieval_rate,
            'answer': record.answer,
            'found_documents': record.found_documents
        }
        return json.dumps(log_record, ensure_ascii=False)


class FileOnlyFilter(logging.Filter):
    def __init__(self, write_flag: str, owner: Handler, *args, **kwargs):
        super().__init__( *args, **kwargs)

        self._write_flag = write_flag
        self._owner = owner

    def filter(self, record: LogRecord) -> bool:
        return getattr(record, self._write_flag, False)


def setup_file_and_stdout_logs(
    logger_name: str,
    log_file_path: str,
    write_flag: Optional[str]=None,
    is_file_jsonl: Optional[bool]=False,
):
    log_file_dir = os.path.dirname(log_file_path)
    os.makedirs(log_file_dir, exist_ok=True)

    logger = getLogger(logger_name)
    logger.propagate = False

    formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stdout_handler = StreamHandler(sys.stdout,)
    stdout_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(log_file_path, maxBytes=1_000_000, encoding='utf-8')
    file_handler.setFormatter(formatter)
    if is_file_jsonl is True:
        file_handler.setFormatter(JSONLFormatter())

    if write_flag:
        file_only_filter = FileOnlyFilter(write_flag, file_handler)
        file_handler.addFilter(file_only_filter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.setLevel(APP_LOG_LEVEL)

    return logger
