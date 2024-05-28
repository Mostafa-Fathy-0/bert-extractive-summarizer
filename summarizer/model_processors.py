import logging
from typing import List, Optional, Type
import numpy as np
from abc import abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer, BertModel, BertTokenizer
from summarizer.bert_parent import BertParent
from summarizer.cluster_features import ClusterFeatures
from summarizer.sentence_handler import SentenceHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProcessor:
    """
    This is the parent BERT Summarizer model. New methods should implement this class.
    """

    def __init__(
        self,
        model: str = 'bert-large-uncased',
        custom_model: Optional[PreTrainedModel] = None,
        custom_tokenizer: Optional[PreTrainedTokenizer] = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):
        """
        Initialize the ModelProcessor.

        :param model: Model name or path for the transformer model.
        :param custom_model: Optional custom pretrained model.
        :param custom_tokenizer: Optional custom tokenizer.
        :param hidden: Hidden layer of the transformer model to use for embeddings.
        :param reduce_option: Method to reduce the embeddings.
        :param sentence_handler: Handler to process sentences.
        :param random_state: Seed for reproducibility.
        """
        np.random.seed(random_state)
        self.model = BertParent(model, custom_model, custom_tokenizer)
        self.hidden = hidden
        self.reduce_option = reduce_option
        self.sentence_handler = sentence_handler
        self.random_state = random_state
        logger.info("ModelProcessor initialized with model: %s", model)

    @abstractmethod
    def run_clusters(
        self,
        content: List[str],
        ratio: float = 0.2,
        algorithm: str = 'kmeans',
        use_first: bool = True
    ) -> List[str]:
        """
        Abstract method to run the clustering algorithm.
        """
        raise NotImplementedError("Must implement run_clusters")

    def run(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans'
    ) -> str:
        """
        Preprocess the sentences, run clusters to find centroids, and combine the sentences.

        :param body: Raw string body to process.
        :param ratio: Ratio of sentences to use.
        :param min_length: Minimum length of sentence candidates.
        :param max_length: Maximum length of sentence candidates.
        :param use_first: Whether to use the first sentence.
        :param algorithm: Clustering algorithm to use.
        :return: Summary sentence.
        """
        sentences = self.sentence_handler(body, min_length, max_length)
        if sentences:
            sentences = self.run_clusters(sentences, ratio, algorithm, use_first)
        return ' '.join(sentences)

    def __call__(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans'
    ) -> str:
        """
        Wrapper for the run function.

        :param body: Raw string body to process.
        :param ratio: Ratio of sentences to use.
        :param min_length: Minimum length of sentence candidates.
        :param max_length: Maximum length of sentence candidates.
        :param use_first: Whether to use the first sentence.
        :param algorithm: Clustering algorithm to use.
        :return: Summary sentence.
        """
        return self.run(body, ratio, min_length, max_length, use_first, algorithm)

class SingleModel(ModelProcessor):
    """
    Deprecated for naming sake.
    """

    def __init__(
        self,
        model: str = 'bert-large-uncased',
        custom_model: Optional[PreTrainedModel] = None,
        custom_tokenizer: Optional[PreTrainedTokenizer] = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):
        super().__init__(
            model=model, custom_model=custom_model, custom_tokenizer=custom_tokenizer,
            hidden=hidden, reduce_option=reduce_option,
            sentence_handler=sentence_handler, random_state=random_state
        )

    def run_clusters(
        self,
        content: List[str],
        ratio: float = 0.2,
        algorithm: str = 'kmeans',
        use_first: bool = True
    ) -> List[str]:
        hidden = self.model(content, self.hidden, self.reduce_option)
        hidden_args = ClusterFeatures(hidden, algorithm, random_state=self.random_state).cluster(ratio)

        if use_first and hidden_args and hidden_args[0] != 0:
            hidden_args.insert(0, 0)

        return [content[j] for j in hidden_args]

class Summarizer(SingleModel):
    """
    This is the main BERT Summarizer class.
    """

    def __init__(
        self,
        model: str = 'bert-large-uncased',
        custom_model: Optional[PreTrainedModel] = None,
        custom_tokenizer: Optional[PreTrainedTokenizer] = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):
        super().__init__(
            model=model, custom_model=custom_model, custom_tokenizer=custom_tokenizer,
            hidden=hidden, reduce_option=reduce_option,
            sentence_handler=sentence_handler, random_state=random_state
        )

class TransformerSummarizer(SingleModel):
    """
    Class for handling various transformer-based models.
    """

    MODEL_DICT = {
        'Bert': (BertModel, BertTokenizer),
        'OpenAIGPT': (OpenAIGPTModel, OpenAIGPTTokenizer),
        'GPT2': (GPT2Model, GPT2Tokenizer),
        'CTRL': (CTRLModel, CTRLTokenizer),
        'TransfoXL': (TransfoXLModel, TransfoXLTokenizer),
        'XLNet': (XLNetModel, XLNetTokenizer),
        'XLM': (XLMModel, XLMTokenizer),
        'DistilBert': (DistilBertModel, DistilBertTokenizer),
    }

    def __init__(
        self,
        transformer_type: str = 'Bert',
        transformer_model_key: str = 'bert-base-uncased',
        transformer_tokenizer_key: Optional[str] = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):
        try:
            self.MODEL_DICT.update({
                'Roberta': (RobertaModel, RobertaTokenizer),
                'Albert': (AlbertModel, AlbertTokenizer),
                'Camembert': (CamembertModel, CamembertTokenizer)
            })
        except Exception as e:
            logger.warning("Could not update MODEL_DICT: %s", str(e))

        model_cls, tokenizer_cls = self.MODEL_DICT[transformer_type]
        model = model_cls.from_pretrained(transformer_model_key, output_hidden_states=True)
        tokenizer = tokenizer_cls.from_pretrained(
            transformer_tokenizer_key if transformer_tokenizer_key else transformer_model_key
        )
        super().__init__(
            None, model, tokenizer, hidden, reduce_option, sentence_handler, random_state
        )
        logger.info("TransformerSummarizer initialized with model type: %s", transformer_type)
