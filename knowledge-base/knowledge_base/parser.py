import logging
from dataclasses import dataclass
from typing import Iterable

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, eq=False)
class ArticleReference:
    reference: str
    title: str


def parse_articles_source(raw_html: str, articles_selector: str) -> Iterable[str]:
    bs = BeautifulSoup(raw_html, features='lxml')

    for a_tag in bs.select(articles_selector):
        href = a_tag.get('href')
        title = a_tag.get('title')
        logger.info(f'{title} ---> {href}')

        yield ArticleReference(reference=href, title=title)
