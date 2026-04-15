import logging
import requests
from typing import Generator, Iterable

logger = logging.getLogger(__name__)


def get_articles_sources(base_url: str) -> Iterable[str]:
    articles_urls = _get_articles_urls(base_url)
    for articles_url in articles_urls:
        articles_source_html = _get_articles_sources_html(articles_url)
        is_finished = articles_source_html is None
        articles_urls.send(is_finished)

        yield articles_source_html


def get_article_html(url: str) -> str | None:
    try:
        resp = requests.get(url)
    except requests.RequestException as ex:
        logger.error(f'{url} reading error: {ex}')
        return None

    if resp.status_code >= 400:
        logger.info(f'{get_article_html.__name__}: {ex}')
        return None

    return resp.content


def _get_articles_urls(base_url: str) -> Generator[str, bool, None]:
    n = 1
    is_last_page = False

    while not is_last_page:
        is_last_page = yield f'{base_url}/{n}/'
        yield
        n += 1


def _get_articles_sources_html(url: str) -> str | None:
    logger.info(f'{_get_articles_sources_html.__name__}: {url}')

    try:
        resp = requests.get(url)
    except requests.RequestException as ex:
        logger.error(f'{url} reading error: {ex}')
        return None

    if resp.status_code == 404:
        logger.info(f'{_get_articles_sources_html.__name__}: processing finished on {url}')
        return None

    logger.info(f'{_get_articles_sources_html.__name__}: html received')
    return resp.content
