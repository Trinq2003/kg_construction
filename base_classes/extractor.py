from abc import ABC, abstractmethod
from typing import Dict, List

from configuration.configurations import ERExtractorConfiguration
from base_classes.llm import AbstractLanguageModel
from nlp.paragraph_processor import ParagraphProcessor


class Extractor(ABC):
    def __init__(self, llm: AbstractLanguageModel, config: ERExtractorConfiguration, paragraph_nlp_processor: ParagraphProcessor) -> None:
        """
        Initialize the entity extractor with an LLM and configuration.

        :param llm: The language model used for entity extraction.
        :param config: The configuration object for the entity extractor.
        """
        self._config = config
        self._llm = llm
        self._paragraph_nlp_processor = paragraph_nlp_processor
    @abstractmethod
    def _extraction_llm_message(self, text: str) -> List[Dict[str, str]]:
        pass