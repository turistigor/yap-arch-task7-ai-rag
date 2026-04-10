import logging
import os
from typing import Iterator

from langchain_community.document_loaders import TextLoader
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BaseChatPromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings

import rag_bot.settings as st
import rag_bot.consts as consts
from rag_bot.logs import APP_LOG_LEVEL
from rag_bot.security import filter_chunks_before_prompt

logging.basicConfig(level=APP_LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_rag_chain(
    kdb_path: str,
    vdb_path: str,
    llm_data: st.LLMData,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData,
    mode: str,
) -> Runnable:

    retriever = create_retriever(kdb_path, vdb_path, embeddings_data, retriever_data)
    llm = _create_llm(llm_data)
    prompt_template = _create_prompt_template(consts.RagMode(mode))

    if mode in {consts.RagMode.SECURITY, consts.RagMode.ALL}:
        return (
            {
                "context": retriever,
                "input": RunnablePassthrough()
            }
            | RunnableLambda(filter_chunks_before_prompt)
            | prompt_template
            | llm
            | StrOutputParser()
        )
    
    return (
        {
            "context": retriever,
            "input": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )


def create_retriever(
    kdb_path: str,
    vdb_path: str,
    embeddings_data: st.EmbeddingsModelData,
    retriever_data: st.RetrieverData,
) -> VectorStoreRetriever:

    vector_db = get_vdb(kdb_path, vdb_path, embeddings_data, retriever_data.vdb)
    return vector_db.as_retriever(
        search_type=retriever_data.search_type,
        search_kwargs={"k": retriever_data.docs_count},
        lambda_mult=retriever_data.lambda_mult,
    )


def _load_documents(kdb_path: str) -> Iterator[Document]:
    documents = []

    for _, _, file_names in os.walk(kdb_path):
        for file_name in file_names:
            file_path = os.path.join(kdb_path, file_name)
            logger.debug(file_path)
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

    logger.app_info(f'Downloaded: {len(documents)}')
    return documents


def _split_documents(
    documents: Iterator[Document], chunk_size: int, chunk_overlap: int,
) -> Iterator[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.app_info(f'Chunks: {len(chunks)}')

    return chunks


def _create_embeddings(embeddings_model: str) -> Embeddings:
    if embeddings_model == consts.ALL_MINI_LM_L6_V2:
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif embeddings_model == consts.ALL_MINI_LM_L6_V2_ST:
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif embeddings_model == consts.ALL_MINI_LM_L6_V2_ST:
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif embeddings_model == consts.PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2:
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif embeddings_model == consts.PARAPHRASE_MULTILINGUAL_MINILM_L12_V2:
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif embeddings_model == consts.MULTILINGUAL_E5_SMALL_IF:
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif embeddings_model == consts.MULTILINGUAL_E5_SMALL_QL:
        embeddings = OllamaEmbeddings(model=embeddings_model)
    elif embeddings_model == consts.BGE_M3:
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif embeddings_model == consts.ALL_MINILM:
        embeddings = OllamaEmbeddings(model=embeddings_model)
    elif embeddings_model == consts.MXBAI_EMBED_LARGE:
        embeddings = OllamaEmbeddings(model=embeddings_model)
    elif embeddings_model == consts.QWEN3_EMBEDDING_4B:
        embeddings = OllamaEmbeddings(model=embeddings_model)
    elif embeddings_model == consts.NOMIC_EMBED_TEXT:
        embeddings = OllamaEmbeddings(model=embeddings_model)
    else:
        raise ValueError(f'{embeddings_model} is not supported')

    logger.app_info('Embeddings created')
    return embeddings


def get_vdb(
    kdb_path: str,
    vdb_path: str,
    embeddings_data: st.EmbeddingsModelData,
    vdb_name: str,
) -> VectorStore:

    embeddings = _create_embeddings(embeddings_data.name)

    if os.path.exists(vdb_path) and os.listdir(vdb_path):
        vector_db = _get_vdb_obj(vdb_path, embeddings, vdb_name)
        logger.app_info("Existing vector db loaded")
    else:
        documents = _load_documents(kdb_path)
        chunks = _split_documents(
            documents, embeddings_data.chunk_size, embeddings_data.chunk_overlap,
        )
        vector_db = _create_vdb(vdb_path, chunks, embeddings, vdb_name)
        logger.app_info("Vector db created")

    return vector_db


def _create_vdb(
    vdb_path: str, chunks: Iterator[Document], embeddings: Embeddings, vdb_name: str,
) -> VectorStore:
    if vdb_name == consts.CHROMA_DB:
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=vdb_path,
        )
    else:
        raise ValueError(f'{vdb_name} is not supported')


def _get_vdb_obj(vdb_path: str, embeddings: Embeddings, vdb_name: str) -> VectorStore:
    if vdb_name == consts.CHROMA_DB:
        return Chroma(
            persist_directory=vdb_path,
            embedding_function=embeddings,
        )
    else:
        raise ValueError(f'{vdb_name} is not supported')


def _create_llm(llm_data: st.LLMData) -> BaseChatModel:
    return ChatOllama(
        model=llm_data.name,
        base_url=llm_data.base_url,
        temperature=llm_data.temperature,
        num_ctx=llm_data.num_ctx,
        num_thread=llm_data.num_thread,
    )


def _create_prompt_template(mode: consts.RagMode) -> BaseChatPromptTemplate:
    if mode == consts.RagMode.MINIMAL:
        return ChatPromptTemplate.from_messages([
            ('system', consts.SYSTEM_CONTEXT),
            ('human', 'Контекст: {context}\n\n Вопрос: {input}'),
        ])
    elif mode == consts.RagMode.FEW_SHOT:
        return ChatPromptTemplate.from_messages([
            ('system', consts.SYSTEM_CONTEXT),
            consts.few_shot_template,
            ('human', 'Контекст: {context}\n\n Вопрос: {input}'),
        ])
    elif mode == consts.RagMode.COT:
        return ChatPromptTemplate.from_messages([
            ('system', consts.SYSTEM_CONTEXT_COT),
            ('human', 'Контекст: {context}\n\n Вопрос: {input}'),
        ])
    elif mode == consts.RagMode.SECURITY:
        return ChatPromptTemplate.from_messages([
            ('system', consts.SYSTEM_CONTEXT_SECURE),
            ('human', 'Контекст: {context}\n\n Вопрос: {input}'),
        ])
    elif mode == consts.RagMode.ALL:
        return ChatPromptTemplate.from_messages([
            ('system', consts.SYSTEM_CONTEXT_FULL),
            consts.few_shot_template,
            ('human', 'Контекст: {context}\n\n Вопрос: {input}'),
        ])

    raise ValueError(f'RAG mode {mode} is not supported now.')
