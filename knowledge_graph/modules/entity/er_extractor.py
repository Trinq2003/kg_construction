from asyncio import current_task
import json
import logging
from typing import Dict, List, Optional
import yaml

from knowledge_graph.modules.node import ContentNode
from exception.entity_exception import EntityTypeListIsEmptyError
from configuration.configurations import ERExtractorConfiguration
from base_classes.llm import AbstractLanguageModel
from base_classes.extractor import Extractor
from base_classes.embedding_model import AbstractEmbeddingModel
from nlp.paragraph_processor import ParagraphProcessor

class ERExtractor(Extractor):
    _entity_types: List[str] = Optional[List[str]]
    emb_model: AbstractEmbeddingModel = None
    def __init__(self, extraction_llm: AbstractLanguageModel, config: ERExtractorConfiguration, paragraph_nlp_processor: ParagraphProcessor) -> None:
        """
        Initialize the entity extractor with an LLM and configuration.

        :param llm: The language model used for entity extraction.
        :param config: The configuration object for the entity extractor.
        """
        super().__init__(llm=extraction_llm, config=config, paragraph_nlp_processor=paragraph_nlp_processor)

    def _load_paragraph(self, paragraph: str) -> None:
        """
        Load a paragraph for entity extraction.

        :param paragraph: The paragraph to load.
        """
        self._paragraph_nlp_processor.load_paragraph(paragraph)
        self._entity_types = self._paragraph_nlp_processor.get_entity_type()
        self._predicates = self._paragraph_nlp_processor.get_predicate()

    def load_embedding_model(self, emb_model: AbstractEmbeddingModel) -> None:
        """
        Load an embedding model for the entity extractor.

        :param emb_model: The embedding model to load.
        """
        self.emb_model = emb_model
        
    def load_yaml_prompt(self):
        """
        Load and parse the YAML prompt template.
        """
        with open(self._config.yaml_prompt_path, 'r', encoding='utf-8') as file:
            self.prompt_data = yaml.safe_load(file)

    def _extraction_llm_message(self, text: str, few_shot: bool = True) -> List[Dict[str, str]]:
        """
        Generates an LLM extraction message, optionally including few-shot examples.
        
        :param text: The input text to process.
        :param few_shot: Whether to include few-shot examples in the prompt.
        :return: A structured LLM message.
        """
        self._load_paragraph(text)
        entity_list_str = ", ".join(self._entity_types)
        predicate_list_str = ", ".join(self._predicates)

        # Extract YAML sections
        system_message = self.prompt_data["ER_EXTRACTION_PROMPT"]["SYTEM_MESSAGE"]
        execution_protocol = self.prompt_data["ER_EXTRACTION_PROMPT"]["EXECUTION_PROTOCOL"]
        output_requirements = self.prompt_data["ER_EXTRACTION_PROMPT"]["OUTPUT_REQUIREMENTS"]
        constraints = self.prompt_data["ER_EXTRACTION_PROMPT"]["CONSTRAINTS"]
        examples = self.prompt_data["ER_EXTRACTION_PROMPT"]["EXAMPLES"] if few_shot else ""

        current_task = self.prompt_data["CURRENT_TASK"].format(entity_list_str=entity_list_str, predicates=predicate_list_str, text=text)
        # Construct user prompt dynamically
        user_prompt = f"""
        {execution_protocol}
        {output_requirements}
        {constraints}
        {examples if few_shot else ""}
        {current_task}        
        """

        extraction_message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        return extraction_message
    
    def _extract_entities_and_relationships_from_given_content(self, text: str, num_responses=1) -> Dict[str, List[Dict]]:
        """
        Extract entities from the given text using the LLM.

        :param text: The input text for entity extraction.
        :return: A dictionary where keys are entity types and values are lists of extracted entities.
        """
        extraction_message = self._extraction_llm_message(text)
        
        raw_responses = self._llm.chat(extraction_message, num_responses=num_responses)
        responses = self._llm.get_response_texts(raw_responses)
        responses_parts = [response.strip().split("\n\n") for response in responses]

        raw_extracted_entities: list = []
        raw_extracted_relationships: list = []
        
        try:
            extraction_data = [[json.loads(response_part[0]), json.loads(response_part[1])] for response_part in responses_parts]
            for data in extraction_data:
                raw_extracted_entities.extend(data[0])
                raw_extracted_relationships.extend(data[1])

        except ValueError as e:
            logging.error(f"Error parsing JSON response: {e}")
        

        return {"raw_entities": raw_extracted_entities, "raw_relationships": raw_extracted_relationships}

    def _refine_extracted_result(self, raw_result: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        TODO: rewrite this function to combine description instead of filtering out entities that do not match the specified entity types.
        Refine the extracted entities and relationships by filtering out entities that do not match the specified entity types.

        :param raw_result: The raw extracted entities and relationships.
        :return: The refined entities and relationships.
        """
        raw_entities = raw_result['raw_entities']
        raw_relationships = raw_result['raw_relationships']

        if len(raw_entities)*len(raw_relationships) == 0:
            return {"entities": [], "relationships": []}
        
        refined_entities = [raw_entities[0]]
        refined_relationships = [raw_relationships[0]]

        for entity in raw_entities[1:]:
            entity_existed = False
            for refined_entity in refined_entities:
                if self.emb_model.is_identical(entity['name'], refined_entity['name']):
                    if self.emb_model.is_identical(entity['description'], refined_entity['description']):
                        entity_existed = True
                        break
            if not entity_existed:
                refined_entities.append(entity)
        
        for relationship in raw_relationships[1:]:
            relationship_existed = False
            for refined_relationship in refined_relationships:
                if self.emb_model.is_identical(relationship['source'], refined_relationship['source']) and self.emb_model.is_identical(relationship['target'], refined_relationship['target']):
                    if self.emb_model.is_identical(relationship['relationship'], refined_relationship['relationship']):
                        relationship_existed = True
                        break
            if not relationship_existed:
                refined_relationships.append(relationship)

        return {"entities": refined_entities, "relationships": refined_relationships}

    def er_extraction(self, content_node: ContentNode, num_trials=1) -> Dict[str, List[str]]:
        raw_extracted_er = self._extract_entities_and_relationships_from_given_content(content_node.content, num_responses=num_trials)
        refined_extracted_er = self._refine_extracted_result(raw_extracted_er)
        return refined_extracted_er
    
