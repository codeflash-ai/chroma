from chromadb.api.types import EmbeddingFunction, Space, Embeddings, Documents
from chromadb.utils.embedding_functions.schemas import validate_config_schema
from typing import List, Dict, Any
import numpy as np


class Text2VecEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    This class is used to generate embeddings for a list of texts using the Text2Vec model.
    """

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        """
        Initialize the Text2VecEmbeddingFunction.

        Args:
            model_name (str, optional): The name of the model to use for text embeddings.
                Defaults to "shibing624/text2vec-base-chinese".
        """
        # Move the import out of the try block for clearer error handling and potentially faster repeated initializations
        import importlib

        if importlib.util.find_spec("text2vec") is None:
            raise ValueError(
                "The text2vec python package is not installed. Please install it with `pip install text2vec`"
            )
        from text2vec import SentenceModel

        self.model_name = model_name
        # SentenceModel loading can be expensive if repeatedly called with the same model_name,
        # so reuse the model across instances if possible. Here we use a class-level cache.
        if not hasattr(self.__class__, "_model_cache"):
            self.__class__._model_cache = {}
        model_cache = self.__class__._model_cache
        if model_name not in model_cache:
            model_cache[model_name] = SentenceModel(model_name_or_path=model_name)
        self._model = model_cache[model_name]

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.

        Args:
            input: Documents or images to generate embeddings for.

        Returns:
            Embeddings for the documents.
        """
        # Text2Vec only works with text documents
        if not all(isinstance(item, str) for item in input):
            raise ValueError("Text2Vec only supports text documents, not images")

        embeddings = self._model.encode(list(input), convert_to_numpy=True)

        # Convert to numpy arrays
        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]

    @staticmethod
    def name() -> str:
        return "text2vec"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction[Documents]":
        model_name = config.get("model_name")

        if model_name is None:
            assert False, "This code should not be reached"

        return Text2VecEmbeddingFunction(model_name=model_name)

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    def validate_config_update(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        # model_name is also used as the identifier for model path if stored locally.
        # Users should be able to change the path if needed, so we should not validate that.
        # e.g. moving file path from /v1/my-model.bin to /v2/my-model.bin
        return

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate the configuration using the JSON schema.

        Args:
            config: Configuration to validate

        Raises:
            ValidationError: If the configuration does not match the schema
        """
        validate_config_schema(config, "text2vec")
