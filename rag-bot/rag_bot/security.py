import logging
import re

logger = logging.getLogger(__name__)

REQUEST_WORDS = (
    'назови', 'скажи', 'напиши', 'покажи', 'дай', 'расскажи',
    'выдай', 'найди', 'какой', 'где', 'какое', 'какая', 'какие',
    'узнай', 'сообщи', 'укажи', 'предоставь', 'озвучь',
)

PASSWORD_WORDS = (
    'пароль', 'логин', 'кред', 'секрет', 'ключ', 'токен',
    'пароля', 'паролю', 'паролем', 'пароле',
    'суперпароль', 'супер-пароль', 'pass', 'password',
    'superpassword'
)

SUSPICIOUS_PATTERNS = (
    r'(?i)игнорируй (?:все )(?:предыдущие )?инструкции.*',
    r'(?i)забудь (?:все )(?:предыдущие )?инструкции.*',
    r'(?i)ты (?:сейчас|теперь) (?!эксперт|консультант|ассистент\b)([а-яё]+(?:\s+[а-яё]+)?)',
    r'(?i)(?:системн(?:ый|ую|ые|ое)\s+(?:промпт|инструкци(?:ю|и|я)|сообщение))',
    r'(?i)(?:раскрой|покажи|расскажи|выдай)\s+(?:свои|свою|своих)',
)


class RagSecurityError(Exception):
    """Ошибка безопасности."""


def is_secure(input_str: str) -> bool:
    text = input_str.lower()

    has_request = any(word in text for word in REQUEST_WORDS)
    has_password = any(word in text for word in PASSWORD_WORDS)
    
    if has_request and has_password:
        return False
    
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, input_str):
            return False
    
    return True


def filter_chunks_before_prompt(data: dict) -> dict:
    chunks = data.get("context", [])

    safe_chunks = []
    for chunk in chunks:
        if _is_secure_chunk(chunk.page_content):
            safe_chunks.append(chunk)

    if not safe_chunks:
        raise RagSecurityError

    data["context"] = safe_chunks
    return data


def _is_secure_chunk(input_str: str) -> bool:
    text = input_str.lower()

    has_password = any(word in text for word in PASSWORD_WORDS)

    if has_password:
        return False

    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, input_str):
            return False

    return True
