import spacy
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

from base_classes.llm import AbstractLanguageModel
from nlp.sentence_processor import SentenceProcessor

class ParagraphProcessor:
    """
    A class for processing paragraphs using NLP techniques.
    - Sentence segmentation using NLP.
    - Pronoun resolution using an LLM through `AbstractLanguageModel`.
    """

    def __init__(self, sentence_nlp_model: SentenceProcessor, language_model: AbstractLanguageModel = None):
        """
        Initialize the NLP model and LLM for pronoun resolution.
        :param model: The spaCy language model to use for sentence segmentation.
        :param language_model: An instance of `AbstractLanguageModel` for pronoun resolution.
        """
        self._sentence_processor: SentenceProcessor = sentence_nlp_model
        self._language_model: AbstractLanguageModel = language_model  # LLM-based inference for pronoun resolution
        self._paragraph: str = ""
        
    @property
    def paragraph(self) -> str:
        return self._paragraph
    
    @property
    def sentences(self) -> List[str]:
        """
        Uses NLP-based sentence segmentation instead of simple string splitting.
        :return: A list of sentences.
        """
        return self._sentences

    def load_paragraph(self, paragraph: str):
        self._paragraph = paragraph
        doc = self._sentence_processor.nlp(self._paragraph)
        self._sentences = [sent.text.strip() for sent in doc.sents]
    
    def get_entity_type(self) -> List[str]:
        """
        Extracts entity types from the paragraph.
        """
        entity_types = []
        for sent in self._sentences:
            self._sentence_processor.load_sentence(sent)
            entity_types.extend(self._sentence_processor.get_entity_type())
        return [entity.upper() for entity in entity_types]
    
    def get_predicate(self) -> List[str]:
        """
        Extracts predicates from the paragraph.
        """
        predicates = []
        for sent in self._sentences:
            self._sentence_processor.load_sentence(sent)
            predicates.extend(self._sentence_processor.get_predicate())
        return [predicate.upper() for predicate in predicates]

    def resolve_pronouns(self, paragraph: str) -> str:
        """
        Resolves personal pronouns using an LLM via `AbstractLanguageModel`.

        Example:
        Input: "My name is Robert. I am talking with Mr. Fisher. He said: 'I am envious of you.'"
        Output: "My name is Robert. Robert is talking with Mr. Fisher. Mr. Fisher said: 'Mr. Fisher is envious of Robert.'"

        :param paragraph: The input paragraph.
        :return: A modified paragraph with pronouns replaced by their referenced named entities.
        """
        if not self._language_model:
            raise ValueError("Language model not provided for pronoun resolution.")

        # Prepare LLM prompt for coreference resolution
        prompt = (
            "You are an AI specializing in coreference resolution. "
            "Replace all personal pronouns (I, he, she, they...) with their referenced named entities while maintaining grammatical correctness.\n\n"
            "Example Input: 'My name is Robert. I am talking with Mr. Fisher. He said: I am envious of you.'\n"
            "Expected Output: 'Robert's name is Robert. Robert is talking with Mr. Fisher. Mr. Fisher said: Mr. Fisher is envious of Robert.'\n\n"
            f"Rewrite this paragraph:\n\n{paragraph}\n\nRewritten Text:"
        )

        # Query the LLM
        response = self.language_model.query(prompt, num_responses=1)

        # Extract response text
        resolved_text = self.language_model.get_response_texts(response)[0]
        return resolved_text
