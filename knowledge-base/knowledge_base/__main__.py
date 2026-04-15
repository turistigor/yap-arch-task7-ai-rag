import logging
import os
import sys
from enum import StrEnum, auto

from knowledge_base.creator import create_knowledge_base
from knowledge_base.replacer import Replacer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTICLES_LIST_URL_BASE = 'https://marathonec.ru/topics/running/training/page'
KNOWLEDGE_BASE_DIR = 'data'
KNOWLEDGE_BASE_CHANGED_DIR = 'data_replaced'
ARTICLES_SELECTOR = '.td-container.td-category-container a.td-image-wrap[rel="bookmark"]'
TERMS_PATH = 'terms_map.json'


class Operations(StrEnum):
    CREATE = auto()
    REPLACE = auto()
    UPDATE = auto()
    ALL = auto()


ARG_ERROR_MESSAGE = f'Укажите допустимую операцию:\n' \
    f' - {Operations.CREATE} - создание базы знаний\n' \
    f' - {Operations.REPLACE} - замена терминов\n'


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.error(ARG_ERROR_MESSAGE)
        exit()

    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    os.makedirs(KNOWLEDGE_BASE_CHANGED_DIR, exist_ok=True)

    op = sys.argv[1]

    if op == Operations.CREATE:
        create_knowledge_base(
            ARTICLES_LIST_URL_BASE, KNOWLEDGE_BASE_DIR, ARTICLES_SELECTOR,
        )
    elif op == Operations.REPLACE:
        replacer = Replacer(
            KNOWLEDGE_BASE_DIR, KNOWLEDGE_BASE_CHANGED_DIR, TERMS_PATH
        )
        replacer.replace_terms()
    elif op == Operations.ALL:
        create_knowledge_base(
            ARTICLES_LIST_URL_BASE, KNOWLEDGE_BASE_DIR, ARTICLES_SELECTOR,
        )
        replacer = Replacer(
            KNOWLEDGE_BASE_DIR, KNOWLEDGE_BASE_CHANGED_DIR, TERMS_PATH
        )
        replacer.replace_terms()
    else:
        logger.error(ARG_ERROR_MESSAGE)
