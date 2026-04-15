import math
from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Iterable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from sentence_transformers import SentenceTransformer, util


from rag_bot.logs import setup_file_and_stdout_logs
from rag_bot.golden_questions import (
    Question, complex_questions, simple_questions, heating_questions,
    unanswered_questions,
)
from rag_bot.rag_creator import create_embeddings

now = datetime.now().strftime('%Y_%m_%d-%H:%M:%S')
FULL_LOG_PATH = f'./logs/{now}-test_rag_bot.log'
FILTER_FLAG_NAME = 'rag_test'
logger = setup_file_and_stdout_logs(
    __name__, FULL_LOG_PATH, FILTER_FLAG_NAME, is_file_jsonl=True,
)

@dataclass(frozen=True, kw_only=True, eq=False)
class AnswerMetrics:
    sem_sim: float         # semantic similarity: cosine_similarity -> [0 - 1.0], 0 - different, 1 - equal
    retrieval_rate: float  # percent of the found chunks in golden chunks -> [0 - 1.0], 0 - np one, 1 - all
    len_diff: float        # length difference: abs(len_difference) / len(golden answer) -> [0 - math.inf]
    found_docs: set[str] = field(default_factory=set)


THRESHOLDS = AnswerMetrics(
    sem_sim=0.7,
    len_diff=0.6,
    retrieval_rate=0.4,
)


def test_vdb(retriever: BaseRetriever):
    logger.app_info('Vector db index test question')
    docs = _find_question_documents(retriever, heating_questions[0])
    for doc in docs:
        print('\nFound chunk:')
        print(f'{doc.id=}')
        print(f'{doc.metadata['source']=}')
        print(f'{doc.page_content=}')


def test_retriever(retriever: BaseRetriever):
    logger.app_info('Retriever test started...\n')

    logger.app_info('Simple questions:')
    _find_documents(retriever, simple_questions)

    logger.app_info('Complex questions:')
    _find_documents(retriever, complex_questions)


def test_rag_bot(rag_chain: Runnable):
    sim_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    logger.app_info('Heating questions:')
    _ask_questions(rag_chain, heating_questions, sim_model)

    logger.app_info('Retriever test started...\n', )

    result_map = {}
    start = datetime.now()

    logger.app_info('Simple questions:')
    result = _ask_questions(rag_chain, simple_questions, sim_model)
    result_map.update(result)

    logger.app_info('Complex questions:')
    result = _ask_questions(rag_chain, complex_questions, sim_model)
    result_map.update(result)

    logger.app_info('Unanswered questions:')
    result = _ask_questions(rag_chain, unanswered_questions, sim_model)
    result_map.update(result)

    elapsed = datetime.now() - start
    logger.app_info(f'Rag bot test duration, min: {(elapsed.total_seconds() / 60):.2f}')
    logger.app_info(_get_test_summary(result_map))


def _get_test_summary(result_map: dict[str, bool]) -> str:
    result_str = 'Test summary:\n'
    for question, good_answer in result_map.items():
        result_str += f'\t{question}: {'OK' if good_answer else 'NOT OK'}\n'
    return result_str


class RagChainMetricsCollector(BaseCallbackHandler):
    def __init__(self):
        self.retrieved_chunks = []
    
    def on_retriever_end(self, retrieved_chunks, **kwargs):
        self.retrieved_chunks = retrieved_chunks


def _ask_questions(
    rag_chain: Runnable, questions: Iterable[Question], sim_model: SentenceTransformer,
) -> dict[str, bool]:
    
    result = {}
    for question in questions:
        logger.app_info(question.text)
        collector = RagChainMetricsCollector()

        try:
            answer = rag_chain.invoke(question.text, config={"callbacks": [collector,]})
        except Exception as ex:
            logger.error(ex)
        else:
            logger.app_info(answer)

        metrics = _check_answer(question, answer, sim_model, collector.retrieved_chunks)
        is_good = _is_answer_good(metrics)

        msg, extra = _get_question_test_results(question.text, answer, metrics, is_good)
        logger.app_info(msg, extra=extra)

        result[question.text] = is_good

    return result


def _get_question_test_results(
    question: str,
    answer: str,
    metrics: AnswerMetrics,
    answer_is_good: bool,
) -> tuple[str, dict]:

    result_str = (
        f'Metrics: \n'
        f'\tsemantic similarity: {metrics.sem_sim:.2f} (≥{THRESHOLDS.sem_sim:.2f})\n'
        f'\tlength difference  : {metrics.len_diff:.2f} (≤{THRESHOLDS.len_diff:.2f})\n'
        f'\tretrieval rate     : {metrics.retrieval_rate:.2f} (≥{THRESHOLDS.retrieval_rate:.2f})\n'
        f'\tTotal              : {"OK" if answer_is_good is True else "NOT OK"}'
    )

    result_dict={
        FILTER_FLAG_NAME: True,
        'question': question,
        'answer': answer,
        'answer_is_good': answer_is_good,
        'semantic_similarity': f'{metrics.sem_sim:.2f}',
        'length_difference': f'{metrics.len_diff:.2f}',
        'retrieval_rate': f'{metrics.retrieval_rate:.2f}',
        'found_documents': ', '.join(metrics.found_docs),
    }

    return result_str, result_dict


def _is_answer_good(metrics: AnswerMetrics) -> bool:
    return (
        metrics.sem_sim >= THRESHOLDS.sem_sim and 
        metrics.len_diff <= THRESHOLDS.len_diff and
        metrics.retrieval_rate >= THRESHOLDS.retrieval_rate
    )


def _calc_cosine_similarity(
    expected_answer: str, answer: str, sim_model: SentenceTransformer,
) -> float:
    emb_expected = sim_model.encode(expected_answer, convert_to_tensor=True)
    emb_answer = sim_model.encode(answer, convert_to_tensor=True)
    similarity = util.cos_sim(emb_expected, emb_answer)

    return similarity.item()


def _calc_len_diff(expected_answer: str, answer: str) -> float:
    exp_len = len(expected_answer)
    real_len = len(answer)

    return abs(1 - real_len / exp_len)


def _calc_retrieval_rate(
    expected_chunks: set[str], found_chunks: Iterable[Document] | None,
) -> tuple[float, set[str]]:
    
    if not found_chunks:
        return 0, set()

    if not expected_chunks:
        return 1, set()

    docs_match = 0
    found_docs = set()

    for chunk in found_chunks:
        chunk_doc = os.path.basename(chunk.metadata['source'])
        if chunk_doc in expected_chunks:
            found_docs.add(chunk_doc)
            docs_match += 1

    rate = docs_match / len(found_chunks) if len(found_chunks) else 0

    return rate, found_docs


def _check_answer(
    question: Question,
    answer: str,
    sim_model: SentenceTransformer,
    found_chunks: Iterable[Document],
) -> AnswerMetrics:
    sem_sim = _calc_cosine_similarity(question.answer, answer, sim_model)
    len_diff = _calc_len_diff(question.answer, answer)
    retrieval_rate, found_docs = _calc_retrieval_rate(question.answer_docs, found_chunks)

    return AnswerMetrics(
        sem_sim=sem_sim,
        len_diff=len_diff,
        retrieval_rate=retrieval_rate,
        found_docs=found_docs,
    )


def _find_documents(retriever: BaseRetriever, questions: Iterable[Question]):
    found_percents = []
    for question in questions:
        docs = _find_question_documents(retriever, question)
        found_percent, found_docs = _calc_retrieval_rate(question.answer_docs, docs)
        for doc in found_docs:
            logger.app_info(f'{doc} - FOUND')
        logger.app_info(f'Found docs percent: {100 * found_percent:.2f}')

        found_percents.append(100 * found_percent)
        print('\n')

    average_percent = sum(found_percents) / len(found_percents)
    logger.app_info(f'Average found percent: {average_percent:.2f}')


def _find_question_documents(retriever: BaseRetriever, question: Question) -> Iterable[Document]:
    logger.app_info(question.text)

    try:
        return retriever.invoke(question.text)
    except Exception as ex:
        logger.error(ex)
