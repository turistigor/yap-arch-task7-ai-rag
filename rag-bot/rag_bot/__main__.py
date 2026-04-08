from enum import StrEnum, auto
import logging
import os
import sys

from dotenv import load_dotenv

import rag_bot.settings as st
import rag_bot.consts as consts
from rag_bot.actions import create_vdb, launch_bot, test_embeddings, test_rag_bot, test_vdb
from rag_bot.logs import APP_LOG_LEVEL

logging.basicConfig(level=APP_LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

KDB_PATH = '../knowledge-base/data_replaced'
VDB_PATH = '../vdb'


def get_settings(
    embeddings_model: str, vdb: str, llm: str,
) -> tuple:
    llm_data = st.LLMData(
        name=llm,
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
        name=embeddings_model,
        chunk_size=500,
        chunk_overlap=200,
        num_ctx=8192,
        num_gpu=0,
        num_thread=8,
    )
    retriever_data = st.RetrieverData(
        vdb=vdb,
        search_type=consts.MMR_SEARCH_TYPE,
        docs_count=5,
        lambda_mult=0,
    )

    return llm_data, embeddings_data, retriever_data


class Operations(StrEnum):
    RUN_BOT=auto()
    CREATE_VDB=auto()
    TEST_VDB=auto()
    TEST_EMBEDDINGS=auto()
    TEST_RAG_BOT=auto()


ARG_ERROR_MESSAGE = f'Укажите допустимую операцию:\n' \
    f' - {Operations.RUN_BOT} - запуск бота\n' \
    f' - {Operations.TEST_EMBEDDINGS} - тестирование модели эмбеддингов\n'


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc < 2:
        logger.error('Укажите тип операции')
        exit()
    else:
        operation = sys.argv[1]

        embeddings_model = sys.argv[2] if argc >= 3 else consts.MULTILINGUAL_E5_SMALL
        vdb_name = sys.argv[3] if argc >= 4 else consts.CHROMA_DB
        llm_name = sys.argv[4] if argc >= 5 else consts.QWEN_2_5_3B
        rag_mode = sys.argv[5] if argc >= 6 else consts.RagMode

    os.makedirs(VDB_PATH, exist_ok=True)
    llm_data, embeddings_data, retriever_data = get_settings(embeddings_model, vdb_name, llm_name)

    if operation == Operations.RUN_BOT:
        launch_bot(KDB_PATH, VDB_PATH, llm_data, embeddings_data, retriever_data, rag_mode)
    elif operation == Operations.CREATE_VDB:
        create_vdb(KDB_PATH, VDB_PATH, embeddings_data, retriever_data)
    elif operation == Operations.TEST_VDB:
        test_vdb(KDB_PATH, VDB_PATH, embeddings_data, retriever_data)
    elif operation == Operations.TEST_EMBEDDINGS:
        test_embeddings(KDB_PATH, VDB_PATH, embeddings_data, retriever_data)
    elif operation == Operations.TEST_RAG_BOT:
        test_rag_bot(KDB_PATH, VDB_PATH, llm_data, embeddings_data, retriever_data, rag_mode)
    else:
        logger.error('Input operation code')
