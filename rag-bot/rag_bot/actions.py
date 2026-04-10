import logging
import os
from datetime import datetime

from langchain_core.runnables import Runnable

import rag_bot.settings as st
import rag_bot.tests as tests
from rag_bot.logs import APP_LOG_LEVEL
from rag_bot.rag_creator import create_embeddings, create_rag_chain, create_retriever, get_vdb, get_vdb_obj
from rag_bot.rag_bot import run_bot
from rag_bot.index_updater import update_index


logging.basicConfig(level=APP_LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def launch_bot(
    kdb_path: str,
    vdb_path: str,
    llm_data: st.LLMData,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData,
    rag_mode: str,
):
    logger.app_info('Model creating started...')

    rag_chain: Runnable = create_rag_chain(
        kdb_path=kdb_path,
        vdb_path=vdb_path,
        llm_data=llm_data,
        embeddings_data=embeddings_data,
        retriever_data=retriever_data,
        mode=rag_mode,
    )

    logger.app_info('Model has been successfully created...')
    run_bot(rag_chain)


def create_vdb(
    kdb_path: str,
    vdb_path: str,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData,
):

    logger.app_info('Vector db creation started...')
    if os.path.exists(vdb_path) and os.listdir(vdb_path):
        logger.app_info(f'Already exists ({vdb_path})')
        return

    start = datetime.now()
    get_vdb(
        kdb_path=kdb_path,
        vdb_path=vdb_path,
        embeddings_data=embeddings_data,
        vdb_name=retriever_data.vdb,
    )
    elapsed = datetime.now() - start
    logger.app_info(f'Vector db creatred in {(elapsed.total_seconds() / 60):.2f} min')


def update_vdb(
    vdb_path: str, kdb_path: str, vdb_name: str, embeddings_data: st.EmbeddingsModelData,
):
    embeddings = create_embeddings(embeddings_data.name)
    vector_db = get_vdb_obj(vdb_path, embeddings, vdb_name)
    update_index(vector_db, kdb_path, embeddings_data)

def test_vdb(
    kdb_path: str,
    vdb_path: str,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData,
):
    retriever = create_retriever(
        kdb_path=kdb_path,
        vdb_path=vdb_path,
        embeddings_data=embeddings_data,
        retriever_data=retriever_data,
    )
    
    tests.test_vdb(retriever)


def test_embeddings(
    kdb_path: str,
    vdb_path: str,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData
):
    start = datetime.now()
    retriever = create_retriever(
        kdb_path=kdb_path,
        vdb_path=vdb_path,
        embeddings_data=embeddings_data,
        retriever_data=retriever_data,
    )
    elapsed = datetime.now() - start
    logger.app_info(f'\nVDB creation time, min: {elapsed.total_seconds() / 60}')

    tests.test_retriever(retriever)


def test_rag_bot(
    kdb_path: str,
    vdb_path: str,
    llm_data: st.LLMData,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData,
    rag_mode: str,
):
    rag_chain: Runnable = create_rag_chain(
        kdb_path=kdb_path,
        vdb_path=vdb_path,
        llm_data=llm_data,
        embeddings_data=embeddings_data,
        retriever_data=retriever_data,
        mode=rag_mode,
    )

    test_rag_bot(rag_chain)