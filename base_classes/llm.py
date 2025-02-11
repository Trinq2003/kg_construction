from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
import logging

from configuration.llm_inference_configuration import LLMConfiguration

class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """
    _llm_model: Any = None
    def __init__(self, llm_config: LLMConfiguration) -> None:
        """
        Initialize the AbstractLanguageModel instance with configuration, model details, and caching options.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config: LLMConfiguration = None

        self.load_config(llm_config)

        self._model_name: str = self._config.model_model_id
        self._temperature: float = self._config.model_temperature
        self._max_tokens: int = self._config.model_max_tokens
        self._cache: bool = self._config.cache_enabled
        self._cache_expiry: int = self._config.cache_cache_expiry

        if self._cache:
            self._response_cache: Dict[str, List[Any]] = {}

        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost: float = 0.0

        self._chat_call_count: int = 0
    
    def load_config(self, llm_config: LLMConfiguration) -> None:
        """
        Load a LLM configuration object.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        self._config = llm_config
        self.logger.debug(f"Config loaded.")
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        Abstract method to load the language model.
        """
        pass

    def clear_cache(self) -> None:
        """
        Clear the response cache.
        """
        self._response_cache.clear()

    def _increment_chat_count(self) -> None:
        """
        Increment the chat call counter.
        """
        self._chat_call_count += 1

    def get_chat_call_count(self) -> int:
        """
        Get the number of times chat() has been called.

        :return: The chat call count.
        :rtype: int
        """
        return self._chat_call_count

    def chat(self, messages: List[Dict], num_responses: int = 1) -> Any:
        """
        Wrapper method for chatting with the language model that automatically increments the call count.
        This method calls the internal _chat method implemented by the subclasses.

        :param messages: The chat messages.
        :type messages: List[Dict]
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        self._increment_chat_count()
        return self._chat(messages, num_responses)

    @abstractmethod
    def _chat(self, messages: List[Dict], num_responses: int = 1) -> Any:
        """
        Abstract method to be implemented by subclasses for chatting with the language model.
        This method should not be called directly; use chat() instead.

        :param messages: The chat messages.
        :type messages: List[Dict]
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        pass

    @abstractmethod
    def query(self, query: str, num_responses: int = 1) -> Any:
        """
        Abstract method to query the language model.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: The number of desired responses.
        :type num_responses: int
        :return: The language model's response(s).
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_response_texts(self, query_responses: Union[List[Any], Any]) -> List[str]:
        """
        Abstract method to extract response texts from the language model's response(s).

        :param query_responses: The responses returned from the language model.
        :type query_responses: Union[List[Any], Any]
        :return: List of textual responses.
        :rtype: List[str]
        """
        pass
