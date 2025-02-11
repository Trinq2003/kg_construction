import spacy
import nltk
from typing import List, Dict, Tuple
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

class SentenceProcessor:
    """
    A class to process sentences for entity and relationship extraction using NLP techniques.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize the NLP model using spaCy.
        :param model: The spaCy language model to use.
        """
        self._nlp = spacy.load(model)
        
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('wordnet')
        
        self._sentence = ""
        self._doc = None
        
    @property
    def nlp(self):
        return self._nlp
    @property
    def sentence(self) -> str:
        return self._sentence
    
    def load_sentence(self, sentence: str):
        self._sentence = sentence
        self._doc = self.nlp(self._sentence)

    def get_proper_noun(self) -> List[str]:
        """
        Extracts proper nouns in complete form.
        """
        proper_nouns = []
        temp_proper_noun = []

        for token in self._doc:
            if token.pos_ == "PROPN":
                temp_proper_noun.append(token.text)
            else:
                if temp_proper_noun:
                    proper_nouns.append(" ".join(temp_proper_noun))
                    temp_proper_noun = []
        
        if temp_proper_noun:
            proper_nouns.append(" ".join(temp_proper_noun))

        return proper_nouns

    def get_common_noun(self) -> List[str]:
        """
        Extracts common nouns and lemmatizes them.
        """
        return [token.lemma_ for token in self._doc if token.pos_ == "NOUN"]

    def get_verb(self) -> List[str]:
        """
        Extracts verbs and lemmatizes them.
        """
        return [token.lemma_ for token in self._doc if token.pos_ == "VERB"]

    def get_entity_type(self) -> List[str]:
        """
        Extracts entity types dynamically using:
        - Named Entity Recognition (NER) for proper nouns.
        - WordNet hypernyms for common nouns.
        - Dependency parsing to refine entity classifications.
        """

        entity_types = set()

        # Use Named Entity Recognition (NER) for Proper Noun Classification
        ner_mappings = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "NORP": "NATIONALITY",
            "FAC": "FACILITY",
            "LOC": "LOCATION",
            "EVENT": "EVENT",
            "PRODUCT": "PRODUCT",
            "LAW": "LAW",
            "WORK_OF_ART": "WORK",
            "LANGUAGE": "LANGUAGE",
            "MONEY": "MONETARY_VALUE",
            "DATE": "TIME",
            "TIME": "TIME",
            "QUANTITY": "QUANTITY",
            "ORDINAL": "ORDINAL",
            "CARDINAL": "NUMBER",
        }

        for ent in self._doc.ents:
            entity_types.add(ner_mappings.get(ent.label_, ent.label_))

        # Add common nouns
        common_nouns = [noun.upper() for noun in self.get_common_noun(sentence)]
        entity_types.update(common_nouns)
        # for noun in common_nouns:
        #     synsets = wn.synsets(noun, pos=wn.NOUN)
        #     if synsets:
        #         hypernyms = synsets[0].hypernyms()  # Get general category
        #         if hypernyms:
        #             entity_types.add(hypernyms[0].lemmas()[0].name().upper())

        return list(entity_types)

    def get_predicate(self) -> List[str]:
        """
        Extracts relationship types using:
        - Verbs (normalized).
        - Dependency parsing to capture relations and phrasal verbs.
        - Handling of copula verbs ("is", "was", etc.).
        """
        predicates = set()

        for token in self._doc:
            # Extract direct verbs as relationships
            if token.pos_ == "VERB":
                predicates.add(token.lemma_.upper())

            # Handle "be-related" cases (e.g., "is located in", "was a CEO")
            if token.lemma_ == "be":
                for child in token.children:
                    if child.dep_ in ["attr", "acomp", "prep"]:  # Attribute, complement, prepositional
                        predicates.add(f"BE_{child.text.upper()}")

            # Extract phrasal verbs (e.g., "take over", "join in")
            if token.dep_ == "prt":  # Particle verb (phrasal verb component)
                phrase = f"{token.head.lemma_}_{token.lemma_}"
                predicates.add(phrase.upper())

        return list(predicates)

if __name__ == "__main__":
    processor = SentenceProcessor()

    sentence = "Tim Cook, CEO of Apple Inc, announced new partnerships with TSMC in California."
    print("\nProper Nouns:", processor.get_proper_noun(sentence))
    print("\nCommon Nouns (Lemmatized):", processor.get_common_noun(sentence))
    print("\nVerbs (Lemmatized):", processor.get_verb(sentence))

    print("\nEntity Types:", processor.get_entity_type(sentence))
    print("\nRelationship Types:", processor.get_predicate(sentence))
