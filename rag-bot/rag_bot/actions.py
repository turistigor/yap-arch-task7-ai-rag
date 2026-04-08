import logging
import os
from datetime import datetime
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

import rag_bot.settings as st
import rag_bot.tests as tests
from rag_bot.rag_creator import create_rag_chain, create_retriever, get_vdb
from rag_bot.rag_bot import run_bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def launch_bot(
    kdb_path: str,
    vdb_path: str,
    llm_data: st.LLMData,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData,
):
    logger.info('Model creating started...')

    rag_chain: Runnable = create_rag_chain(
        kdb_path=kdb_path,
        vdb_path=vdb_path,
        llm_data=llm_data,
        embeddings_data=embeddings_data,
        retriever_data=retriever_data,
    )

    logger.info('Model has been successfuly created...')
    run_bot(rag_chain)


def create_vdb(
    kdb_path: str,
    vdb_path: str,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData,
):

    logger.info('Vector db creation started...')
    if os.path.exists(vdb_path) and os.listdir(vdb_path):
        logger.info(f'Already exists ({vdb_path})')
        return

    start = datetime.now()
    get_vdb(
        kdb_path=kdb_path,
        vdb_path=vdb_path,
        embeddings_data=embeddings_data,
        vdb_name=retriever_data.vdb,
    )
    elapsed = datetime.now() - start
    logger.info(f'Vector db creatred in {(elapsed.total_seconds() / 60):.2f} min')


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
    logger.info(f'\nVDB creation time, min: {elapsed.total_seconds() / 60}')

    tests.test_retriever(retriever)


def test_rag_bot(
    kdb_path: str,
    vdb_path: str,
    llm_data: st.LLMData,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData,
):
    rag_chain: Runnable = create_rag_chain(
        kdb_path=kdb_path,
        vdb_path=vdb_path,
        llm_data=llm_data,
        embeddings_data=embeddings_data,
        retriever_data=retriever_data,
    )

    test_rag_bot(rag_chain)