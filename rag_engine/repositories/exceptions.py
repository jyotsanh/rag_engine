from rag_engine.exceptions import AppValidationError, ConflictError, NotFoundError


class DocumentNotFoundError(NotFoundError):
    code = "document_not_found"
    default_message = "Document not found"


class DuplicateDocumentError(ConflictError):
    code = "duplicate_document"
    default_message = "A document with the same unique key already exists"


class InvalidCursorError(AppValidationError):
    code = "invalid_cursor"
    default_message = "Pagination cursor is invalid"
