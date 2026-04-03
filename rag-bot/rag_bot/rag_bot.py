import logging

from langchain_core.runnables import Runnable

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_bot(rag_chain: Runnable):
    while True:
        print('\nInput your question:\n')
        input_str = input().strip()

        if _is_exit(input_str.lower()):
            exit(0)
        
        if not input_str:
            continue

        try:
            answer = rag_chain.invoke(input_str)
        except Exception as ex:
            logger.error(ex)
            exit(0)
        else:
            print(answer)


def _is_exit(input_str: str) -> bool:
    return input_str in ('q', 'quit', 'exit')
