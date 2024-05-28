import logging
from typing import List, Optional, Tuple, Type, Union
import torch
import numpy as np
from numpy import ndarray
from transformers import (
    BertModel, BertTokenizer, XLNetModel, XLNetTokenizer, XLMModel, XLMTokenizer,
    DistilBertModel, DistilBertTokenizer, AlbertModel, AlbertTokenizer,
    PreTrainedModel, PreTrainedTokenizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertParent:
    """
    Base handler for BERT models.
    """

    MODELS = {
        'bert-base-uncased': (BertModel, BertTokenizer),
        'bert-large-uncased': (BertModel, BertTokenizer),
        'xlnet-base-cased': (XLNetModel, XLNetTokenizer),
        'xlm-mlm-enfr-1024': (XLMModel, XLMTokenizer),
        'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer),
        'albert-base-v1': (AlbertModel, AlbertTokenizer),
        'albert-large-v1': (AlbertModel, AlbertTokenizer)
    }

    def __init__(
        self,
        model: str,
        custom_model: Optional[PreTrainedModel] = None,
        custom_tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initializes the BertParent with the specified model.

        :param model: The model name or path for the BERT weights.
        :param custom_model: Optional custom pretrained model.
        :param custom_tokenizer: Optional custom tokenizer.
        """
        base_model, base_tokenizer = self.MODELS.get(model, (None, None))

        if custom_model:
            self.model = custom_model
            logger.info("Using custom model.")
        else:
            if base_model is None:
                logger.error("Model name not recognized.")
                raise ValueError("Model name not recognized.")
            self.model = base_model.from_pretrained(model, output_hidden_states=True)
            logger.info("Loaded model %s from pretrained.", model)

        if custom_tokenizer:
            self.tokenizer = custom_tokenizer
            logger.info("Using custom tokenizer.")
        else:
            if base_tokenizer is None:
                logger.error("Tokenizer name not recognized.")
                raise ValueError("Tokenizer name not recognized.")
            self.tokenizer = base_tokenizer.from_pretrained(model)
            logger.info("Loaded tokenizer %s from pretrained.", model)

        self.model.eval()

    def tokenize_input(self, text: str) -> torch.Tensor:
        """
        Tokenizes the text input.

        :param text: Text to tokenize.
        :return: Tokenized text as a torch tensor.
        """
        if not isinstance(text, str):
            logger.error("Input text must be a string.")
            raise ValueError("Input text must be a string.")
        
        tokenized_text = self.tokenizer(text, return_tensors='pt')
        logger.info("Tokenized input text.")
        return tokenized_text['input_ids']

    def extract_embeddings(
        self,
        text: str,
        hidden: int = -2,
        squeeze: bool = False,
        reduce_option: str = 'mean'
    ) -> ndarray:
        """
        Extracts embeddings for the given text.

        :param text: The text to extract embeddings for.
        :param hidden: The hidden layer to use for extraction.
        :param squeeze: Whether to squeeze the outputs.
        :param reduce_option: How to reduce the hidden states (mean, max, median).
        :return: Embeddings as a numpy array.
        """
        tokens_tensor = self.tokenize_input(text)
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            hidden_states = outputs.hidden_states

        layer_output = hidden_states[hidden]

        if reduce_option == 'max':
            pooled = torch.max(layer_output, dim=1).values
        elif reduce_option == 'median':
            pooled = torch.median(layer_output, dim=1).values
        else:
            pooled = torch.mean(layer_output, dim=1)

        if squeeze:
            result = pooled.squeeze().detach().numpy()
        else:
            result = pooled.detach().numpy()

        logger.info("Extracted embeddings with reduce option: %s", reduce_option)
        return result

    def create_matrix(
        self,
        content: List[str],
        hidden: int = -2,
        reduce_option: str = 'mean'
    ) -> ndarray:
        """
        Creates a matrix from the embeddings of the given content.

        :param content: List of sentences.
        :param hidden: Hidden layer to use.
        :param reduce_option: Option to reduce the hidden states (mean, max, median).
        :return: Numpy array matrix of embeddings.
        """
        if not isinstance(content, list) or not all(isinstance(item, str) for item in content):
            logger.error("Content must be a list of strings.")
            raise ValueError("Content must be a list of strings.")

        embeddings = [
            self.extract_embeddings(text, hidden=hidden, reduce_option=reduce_option)
            for text in content
        ]
        matrix = np.array(embeddings)
        logger.info("Created embedding matrix from content.")
        return matrix

    def __call__(
        self,
        content: List[str],
        hidden: int = -2,
        reduce_option: str = 'mean'
    ) -> ndarray:
        """
        Allows the instance to be called as a function to create a matrix from content.

        :param content: List of sentences.
        :param hidden: Hidden layer to use.
        :param reduce_option: Option to reduce the hidden states (mean, max, median).
        :return: Numpy array matrix of embeddings.
        """
        return self.create_matrix(content, hidden, reduce_option)