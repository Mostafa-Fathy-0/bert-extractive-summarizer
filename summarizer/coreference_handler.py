import logging
from typing import List, Type
from spacy.lang.en import English
import neuralcoref
from summarizer.sentence_handler import SentenceHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoreferenceHandler(SentenceHandler):
    """
    A class for handling coreference resolution and sentence processing.
    """

    def __init__(self, language: Type[English] = English, greedyness: float = 0.45):
        """
        Initializes the CoreferenceHandler with the specified language model and greedyness for coreference resolution.

        :param language: The language model to use (default is English).
        :param greedyness: Greedyness parameter for coreference resolution.
        """
        super().__init__(language)
        neuralcoref.add_to_pipe(self.nlp, greedyness=greedyness)
        logger.info("CoreferenceHandler initialized with greedyness: %f", greedyness)

    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences with coreference resolution.

        :param body: The raw string body to process.
        :param min_length: Minimum length that the sentences must be.
        :param max_length: Maximum length that the sentences must fall under.
        :return: A list of sentences that meet the length criteria.
        """
        if not isinstance(body, str):
            logger.error("Input body must be a string.")
            raise ValueError("Input body must be a string.")
        
        if min_length <= 0 or max_length <= 0:
            logger.error("Minimum and maximum lengths must be positive integers.")
            raise ValueError("Minimum and maximum lengths must be positive integers.")
        
        if min_length >= max_length:
            logger.error("Minimum length must be less than maximum length.")
            raise ValueError("Minimum length must be less than maximum length.")

        resolved_text = self.nlp(body)._.coref_resolved
        doc = self.nlp(resolved_text)
        sentences = [sent.text.strip() for sent in doc.sents if max_length > len(sent.text.strip()) > min_length]

        logger.info("Processed %d sentences with coreference resolution.", len(sentences))
        return sentences

    def __call__(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Allows the instance to be called as a function to process sentences.

        :param body: The raw string body to process.
        :param min_length: Minimum length that the sentences must be.
        :param max_length: Maximum length that the sentences must fall under.
        :return: A list of sentences that meet the length criteria.
        """
        return self.process(body, min_length, max_length)