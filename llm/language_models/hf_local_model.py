import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Union

from base_classes.llm import AbstractLanguageModel
from configuration.llm_inference_configuration import LocalLLMConfiguration

class HuggingfaceLocalInference(AbstractLanguageModel):
    """
    A subclass of AbstractLanguageModel for running inference with a locally stored Hugging Face model.
    """
    def __init__(self, llm_config: LocalLLMConfiguration):
        """
        Initialize the HuggingfaceLocalInference model using a local model path.

        :param llm_config: Configuration object containing paths to the local model, tokenizer, and config.
        """
        super().__init__(llm_config)
        
        # Hugging Face specific parameters
        self.__model_local_path: str = self._config.path_local_model
        self.__tokenizer_local_path: str = self._config.path_local_tokenizer
        self.__config_local_path: str = self._config.path_local_config
        self.__cache_local_path: str = self._config.path_local_cache
        self.__device: str = self._config.deployment_device
        self.__tensor_parallel_size: int = self._config.deployment_tensor_parallel_size
        self.__gpu_memory_utilization: float = self._config.deployment_gpu_memory_utilization
        self.__dtype: str = self._config.deployment_dtype
        
        self._load_model()
        
        
    def _load_model(self):

        if not self.__model_local_path or not self.__tokenizer_local_path:
            raise ValueError("Model path and tokenizer path must be specified in the configuration.")

        self._tokenizer = AutoTokenizer.from_pretrained(self.__tokenizer_local_path)
        self._llm_model = AutoModelForCausalLM.from_pretrained(self.__model_local_path).to(self.__device)

        self.logger.info(f"Loaded model from {self.__model_local_path} on device {self.__device}")

    def _chat(self, messages: List[Dict], num_responses: int = 1) -> List[str]:
        """
        Generates a response for a given chat history using the local Hugging Face model.

        :param messages: The chat history in the form of a list of dictionaries.
        :param num_responses: The number of responses to generate.
        :return: A list of generated responses.
        """
        chat_text = self._format_messages(messages)
        return self.query(chat_text, num_responses)

    def query(self, query: str, num_responses: int = 1) -> List[str]:
        """
        Queries the local Hugging Face model and returns generated responses.

        :param query: The input query string.
        :param num_responses: The number of responses to generate.
        :return: A list of generated responses.
        """
        inputs = self._tokenizer(query, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._llm_model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=num_responses,
                do_sample=False
            )

        return self.get_response_texts(outputs)

    def get_response_texts(self, query_responses: Union[List[Any], Any]) -> List[str]:
        """
        Extracts text responses from the model output.

        :param query_responses: The raw output from the language model.
        :return: A list of response strings.
        """
        return [self._tokenizer.decode(output, skip_special_tokens=True) for output in query_responses]

    @staticmethod
    def _format_messages(messages: List[Dict]) -> str:
        """
        Formats chat messages into a single text input.

        :param messages: List of message dictionaries.
        :return: A formatted string for model input.
        """
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
