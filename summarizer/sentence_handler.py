import logging
from typing import List, Type
from spacy.lang.en import English
from spacy.language import Language

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceHandler:
    def __init__(self, language: Type[Language] = English):
        """
        Initializes the SentenceHandler with the specified language model.

        :param language: A spaCy language model class, default is English.
        """
        self.nlp = language()
        if not self.nlp.has_pipe('sentencizer'):
            self.nlp.add_pipe('sentencizer')
        logger.info("SentenceHandler initialized with language: %s", language)

    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process.
        :param min_length: Minimum length that the sentences must be.
        :param max_length: Maximum length that the sentences must fall under.
        :return: A list of sentences that meet the length criteria.
        :raises ValueError: If the body is not a string.
        """
        if not isinstance(body, str):
            logger.error("Invalid input: body must be a string")
            raise ValueError("Input body must be a string")

        doc = self.nlp(body)
        sentences = [sent.text.strip() for sent in doc.sents if min_length < len(sent.text.strip()) < max_length]
        logger.info("Processed %d sentences from the input body", len(sentences))
        return sentences

    def __call__(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Allows the instance to be called as a function to process the content sentences.

        :param body: The raw string body to process.
        :param min_length: Minimum length that the sentences must be.
        :param max_length: Maximum length that the sentences must fall under.
        :return: A list of sentences that meet the length criteria.
        """
        return self.process(body, min_length, max_length)

# Example usage:
if __name__ == "__main__":
    handler = SentenceHandler()
    text = "This is a sentence. This is another longer sentence that should be included. Short. A very very long sentence that exceeds the maximum length limit set in the handler class for processing."
    sentences = handler(text, min_length=10, max_length=50)
    for sentence in sentences:
        print(sentence)
