import trafilatura

from knowledge_base.downloader import get_articles_sources, get_article_html
from knowledge_base.parser import parse_articles_source
from knowledge_base.saver import save_article


def create_knowledge_base(
    articles_list_url_base: str,
    knowledge_base_dir: str,
    articles_selector: str,
):
    for articles_source_html in get_articles_sources(articles_list_url_base):

        if articles_source_html is None:
            break

        articles_sources = parse_articles_source(articles_source_html, articles_selector)
        for ariticle_source in articles_sources:
            article_html = get_article_html(ariticle_source.reference)
            article = trafilatura.extract(article_html)
            save_article(article, ariticle_source.title, knowledge_base_dir)
