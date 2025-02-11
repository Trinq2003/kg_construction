from abc import ABC, abstractmethod
from typing import List, Dict, Union
import os
import time
import backoff
from openai import AzureOpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from configuration.llm_inference_configuration import APILLMConfiguration
from exception.llm_exception import APIKeyIsNotSetError, APIBaseIsNotSetError
from base_classes.llm import AbstractLanguageModel

class AzureGPT(AbstractLanguageModel):
    """
    The AzureGPT class handles interactions with an Azure OpenAI deployed model.
    It inherits from AbstractLanguageModel and implements the necessary methods.
    """

    def __init__(self, llm_config: APILLMConfiguration) -> None:
        """
        Initialize the AzureGPT instance with configuration, model details, and caching options.
        """
        super().__init__(llm_config)

        # Azure specific parameters
        self.__api_key: str = self._config.llm_api_api_key
        self.__api_base: str = self._config.llm_api_api_base
        self.__api_version: str = self._config.llm_api_api_version
        self.__deployment_id: str = self._config.llm_api_deployment_name

        self.prompt_token_cost: float = self._config.cost_prompt_token_cost
        self.response_token_cost: float = self._config.cost_response_token_cost
        
        self.__max_retries: int = self._config.retry_max_retries
        self.__backoff_factor: float = self._config.retry_backoff_factor
        
        self.encryption_enabled: bool = self._config.security_encrypted
        self.trust_strategy: str = self._config.security_trust_strategy
        
        self._load_model()

    def _load_model(self):
        if not self.__api_key:
            raise APIKeyIsNotSetError()
        if not self.__api_base:
            raise APIBaseIsNotSetError()
        
        self.__llm = AzureOpenAI(api_key = self.__api_key, 
                                 azure_endpoint = self.__api_base,
                                 api_version = self.__api_version)


    def query(self, query: str, num_responses: int = 1) -> Union[Dict, List[Dict]]:
        """
        Query the Azure OpenAI GPT model.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the Azure OpenAI model.
        :rtype: Dict
        """
        # Check if the response is in the cache, if caching is enabled and the response is cached return the response without calling the model
        if self._cache and query in self._response_cache:
            return self._response_cache[query]

        if num_responses == 1:
            response = self.chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
                ], num_responses)
        else:
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    assert next_try > 0
                    res = self.chat([{"role": "user", "content": query}], next_try)
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        f"Error in chatgpt: {e}, trying again with {next_try} samples"
                    )
                    time.sleep(1)
                    total_num_attempts -= 1

        if self._cache:
            self._response_cache[query] = response

        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def _chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        """
        Send chat messages to the OpenAI model and retrieves the model's response.
        Implements backoff on OpenAI error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The OpenAI model's response.
        :rtype: ChatCompletion
        """
        response = self.__llm.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            n=num_responses,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0

        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )

        self.logger.info(
            f"Consumed {prompt_tokens_k} prompt tokens and {completion_tokens_k} completion tokens."
            f"Cost for the chat: {self.cost}"
        )
        return response

    def get_response_texts(self, query_response: Union[Dict, List[Dict]]) -> List[str]:
        response_texts = []
    
        for choice in query_response.choices:
            response_texts.append(choice.message.content)
        
        return response_texts
