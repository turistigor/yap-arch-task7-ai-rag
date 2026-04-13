import logging
from getpass import getuser

from langchain_core.runnables import Runnable

from rag_bot.security import is_secure, RagSecurityError

logger = logging.getLogger(__name__)

BOT_NAME = 'БеБо'
BOT_NAME_FULL = 'Беговой Ботаник'

USER = getuser()

WELCOME_STR = \
    f'\nПривет, {USER}!\n' \
    f'Меня зовут {BOT_NAME} ({BOT_NAME_FULL} 🤖), я - эксперт по марафонскому бегу.\n' \
    f'Готов ответить на твои вопросы по бегу и смежным дисциплинам.🏃🏊🚴'

INSECURE_MESSAGE = (
    'Кажется это не спортивный запрос! Он выглядит подозрительным. 🕵️\n'
    'К сожалению я не могу на него ответить...'
)

ACCESS_FORBIDDEN_MESSAGE = (
    'К сожалению я не могу ответить на это вопрос по соображениям конфиденциальности. 👮‍♂️\n'
)


def run_bot(rag_chain: Runnable):
    print(WELCOME_STR)

    while True:
        print(f'\n{USER}:')
        input_str = input().strip()

        if _is_exit(input_str.lower()):
            exit(0)

        if not is_secure(input_str):
            print(f'\n{BOT_NAME}:\n{INSECURE_MESSAGE}')
            continue

        if not input_str:
            continue

        try:
            answer = rag_chain.invoke(input_str)
        except RagSecurityError as ex:
            print(f'\n{BOT_NAME}:\n{ACCESS_FORBIDDEN_MESSAGE}')
        except Exception as ex:
            logger.error(ex)
            exit(0)
        else:
            print(f'\n{BOT_NAME}:\n{answer}')


def _is_exit(input_str: str) -> bool:
    return input_str in {'q', 'quit', 'exit'}
