from rag_engine.exceptions import AppValidationError


class UnsupportedEmbeddingModelError(AppValidationError):
    code = "unsupported_embedding_model"
    default_message = "Embedding model is not supported"


class UnsupportedEmbeddingDimensionsError(AppValidationError):
    code = "unsupported_embedding_dimensions"
    default_message = "Requested embedding dimensions are not supported by this model"
