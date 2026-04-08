from datetime import datetime
import logging
import math
from dataclasses import dataclass
from typing import Iterator, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from rag_bot.logs import APP_LOG_LEVEL

logging.basicConfig(level=APP_LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, eq=False)
class Question:
    text: str
    answer_docs: Optional[Iterator[str]] = None  # наиболее важный для ответа документ


heating_questions = (
    Question(
        text='Какова длина марафона?',
        answer_docs=(),
    ),
)

simple_questions = (
    Question(
        text='Что такое дергунчик?',
        answer_docs=(
            'Что_такое_дергунчик_и_как_его_бегать.txt',
            'Зачем_бегать_интервалы_при_подготовке_к_марафону.txt',
        )
    ),
    Question(
        text='Что такое тыгыдык?',
        answer_docs=(
            'Тыгыдык_при_беге:_каким_он_должен_быть_и_как_его_тренировать.txt',
        )
    ),
    Question(
        text='Что такое туки-тук?',
        answer_docs=(
            'Пульсовые_зоны:_на_каком_пульсе_бегать.txt',
        )
    ),
    Question(
        text='Что такое туки-тук макс?',
        answer_docs=(
            'Пульсовые_зоны:_на_каком_пульсе_бегать.txt',
        )
    ),
    Question(
        text='Что представляет собой тест Медякова?',
        answer_docs=(
            'Тест_Медякова_для_аэробных_видов_спорта:_бег,_плавание,_велосипед.txt',
        )
    ),
)

complex_questions = (
    Question(
        text='Чем дергунчик отличается от интервалов?',
        answer_docs=(
            'Что_такое_дергунчик_и_как_его_бегать.txt',
            'Зачем_бегать_интервалы_при_подготовке_к_марафону.txt',
            '5_интервальных_беговых_тренировок_для_новичков.txt',
        ),
    ),
    Question(
        text='Каким должен быть оптимальный тыгыдык?',
        answer_docs=(
            'Тыгыдык_при_беге:_каким_он_должен_быть_и_как_его_тренировать.txt',
            'Техника_бега_на_длинные_дистанции:_5_основных_правил.txt',
            'Как_пробежать_первый_полумарафон?_12_простых_шагов.txt'
        ),
    ),
    Question(
        text='Что использовать для измерения туки-тук?',
        answer_docs=(
            '«Ваш_первый_марафон»:_8_ключевых_идей_книги_Грете_Вайтц.txt',
        ),
    ),
    Question(
        text='Для чего используется туки-тук макс?',
        answer_docs=(
            'Как_рассчитать_максимальный_пульс_(туки-тук макс).txt',
            'Пульсовые_зоны:_на_каком_пульсе_бегать.txt',
        ),
    ),
    Question(
        text='Перечисли разновидности теста Медякова?',
        answer_docs=(
            'Тест_Медякова_для_аэробных_видов_спорта:_бег,_плавание,_велосипед.txt',
        ),
    ),
)


def test_vdb(retriever: BaseRetriever):
    logger.app_info('Vector db index test question')
    docs = _find_document(retriever, heating_questions[0])
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
    logger.app_info('Heating questions:')
    _ask_questions(rag_chain, heating_questions)

    logger.app_info('Retriever test started...\n')

    start = datetime.now()

    logger.app_info('Simple questions:')
    _ask_questions(rag_chain, simple_questions)

    logger.app_info('Complex questions:')
    _ask_questions(rag_chain, complex_questions)

    elapsed = datetime.now() - start
    logger.app_info(f'\nRag bot test duration, min: {elapsed.total_seconds() / 60}')


def _ask_questions(rag_chain: Runnable, questions: Iterator[Question]):
    for question in questions:
        logger.app_info(question.text)

        try:
            answer = rag_chain.invoke(question.text)
        except Exception as ex:
            logger.error(ex)
        else:
            logger.app_info(answer)


def _find_documents(retriever: BaseRetriever, questions: Iterator[Question]):
    found_percents = []
    for question in questions:
        docs = _find_document(retriever, question)
        found_percent = _analyze_docs(question.answer_docs, docs)
        found_percents.append(found_percent)
        print('\n')
    
    average_percent = sum(found_percents) / len(found_percents)
    logger.app_info(f'Average found percent: {average_percent:.2f}')


def _analyze_docs(expected_docs: Iterator[str], docs: Iterator[Document]) -> float:
    docs_match = 0
    for expected_doc in expected_docs:
        for doc in docs:
            if expected_doc in doc.metadata['source']:
                logger.app_info(f'{expected_doc} - FOUND')
                docs_match += 1
                break

    percentage = 100 * docs_match / len(expected_docs) if len(expected_docs) else math.nan
    logger.app_info(f'Found docs percent: {percentage:.2f}')
    return percentage


def _find_document(retriever: BaseRetriever, question: Question) -> Iterator[Document]:
    logger.app_info(question.text)

    try:
        return retriever.invoke(question.text)
    except Exception as ex:
        logger.error(ex)
