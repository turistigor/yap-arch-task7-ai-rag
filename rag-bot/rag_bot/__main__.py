from datetime import datetime
from enum import StrEnum, auto
import logging
import os
import sys

from langchain_core.runnables import Runnable

import rag_bot.settings as st
import rag_bot.consts as consts
from rag_bot.rag_creator import create_rag_chain, create_retriever
from rag_bot.rag_bot import run_bot
from rag_bot.tests import test_rag_bot, test_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_CHANGED_DIR = '../knowledge-base/data_replaced'
VDB_PATH = '../vdb'


def get_settings() -> tuple:
    llm_data = st.LLMData(
        name=consts.QWEN_2_5_3B,
        base_url=consts.OLAMA_BASE_URL,
        temperature=0.1,
        num_ctx=8192,
        num_gpu=0,
        num_thread=8,
        num_predict=512,
        top_k=40,
        top_p=0.9
    )
    embeddings_data = st.EmbeddingsModelData(
        name=consts.MULTILINGUAL_E5_SMALL,
        chunk_size=500,
        chunk_overlap=200,
        num_ctx=8192,
        num_gpu=0,
        num_thread=8,
    )
    retreiver_data = st.RetrieverData(
        search_type=consts.MMR_SEARCH_TYPE,
        docs_count=5,
        lambda_mult=0,
    )

    return llm_data, embeddings_data, retreiver_data


class Operations(StrEnum):
    RUN_BOT=auto()
    TEST_EMBEDDINGS=auto()
    TEST_RAG_BOT=auto()


ARG_ERROR_MESSAGE = f'Укажите допустимую операцию:\n' \
    f' - {Operations.RUN_BOT} - запуск бота\n' \
    f' - {Operations.TEST_EMBEDDINGS} - тестирование модели эмбеддингов\n'


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.error()
        exit()
    else:
        operation = sys.argv[1]

    os.makedirs(VDB_PATH, exist_ok=True)
    llm_data, embeddings_data, retreiver_data = get_settings()

    if operation == Operations.RUN_BOT:
        logger.info('Model creating started...')

        rag_chain: Runnable = create_rag_chain(
            kdb_path=KNOWLEDGE_BASE_CHANGED_DIR,
            vdb_path=VDB_PATH,
            llm_data=llm_data,
            embeddings_model=embeddings_data,
            retreiver_data=retreiver_data,
        )

        logger.info('Model has been successfuly created...')
        run_bot(rag_chain)

    elif operation == Operations.TEST_EMBEDDINGS:
        start = datetime.now()
        retreiver = create_retriever(
            kdb_path=KNOWLEDGE_BASE_CHANGED_DIR,
            vdb_path=VDB_PATH,
            embeddings_model=embeddings_data,
            retriever_data=retreiver_data,
        )
        elapsed = datetime.now() - start
        logger.info(f'\nVDB creation time, min: {elapsed.total_seconds() / 60}')

        test_retriever(retreiver)

    elif operation == Operations.TEST_RAG_BOT:
        rag_chain: Runnable = create_rag_chain(
            kdb_path=KNOWLEDGE_BASE_CHANGED_DIR,
            vdb_path=VDB_PATH,
            llm_data=llm_data,
            embeddings_model=embeddings_data,
            retreiver_data=retreiver_data,
        )

        test_rag_bot(rag_chain)
    else:
        logger.error('Input operation code')
