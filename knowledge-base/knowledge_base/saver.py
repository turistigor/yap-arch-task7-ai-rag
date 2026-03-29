import os

FILE_EXT = 'txt'

def save_article(text: str, title: str, knowledge_base_dir: str):
    file_name = f'{title.replace(' ', "_")}.{FILE_EXT}'
    file_name = file_name.replace('/', "-")
    path = os.path.join(knowledge_base_dir, file_name)

    with open(path, 'w') as article_file:
        article_file.write(text)
