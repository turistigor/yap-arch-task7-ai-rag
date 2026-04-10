import json
import logging
import os
from typing import Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Replacer:
    def __init__(
        self,
        knowledge_base_dir: str,
        knowledge_base_dir_new: str,
        replaced_terms_path: str
    ):
        self._kdb_path = knowledge_base_dir
        self._kdb_path_new = knowledge_base_dir_new

        with open(replaced_terms_path) as terms_file:
            self._terms = json.load(terms_file)

    def replace_terms(self):
        for _, _, kdb_file_names in os.walk(self._kdb_path):
            for kdb_file_name in kdb_file_names:
                logger.info(kdb_file_name)

                kdb_file_name_new = self._replace_terms(kdb_file_name)
                if kdb_file_name_new != kdb_file_name:
                    logger.debug(f'{kdb_file_name} -> \n{kdb_file_name_new}')

                lines = self._replace_terms_in_file(kdb_file_name)

                kdb_file_path_new = os.path.join(self._kdb_path_new, kdb_file_name_new)
                with open(kdb_file_path_new, '+w') as kdb_file_new:
                    kdb_file_new.writelines(lines)

    def _replace_terms_in_file(self, file_name: str) -> Iterable[str]:
        lines = []
        kdb_file_path = os.path.join(self._kdb_path, file_name)

        with open(kdb_file_path) as kdb_file:
            for kdb_line in kdb_file:
                kdb_line_new = self._replace_terms(kdb_line)
                if kdb_line_new != kdb_line:
                    logger.info(f'{kdb_line} -> {kdb_line_new}' )

                lines.append(kdb_line_new)

        return lines

    def _replace_terms(self, line: str) -> str:
        for term, term_new in self._terms.items():
            if term in line:
                logger.info("!!!")
                line = line.replace(term, term_new)
        return line
