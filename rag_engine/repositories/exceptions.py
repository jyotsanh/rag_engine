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


class TenantNotFoundError(DocumentNotFoundError):
    code = "tenant_not_found"
    default_message = "Tenant not found"


class TenantNameTakenError(ConflictError):
    code = "tenant_name_taken"
    default_message = "A tenant with this name already exists"


class ApiKeyNotFoundError(DocumentNotFoundError):
    code = "api_key_not_found"
    default_message = "API key not found"


class KnowledgeBaseNotFoundError(DocumentNotFoundError):
    code = "knowledge_base_not_found"
    default_message = "Knowledge base not found"


class KnowledgeBaseNameTakenError(ConflictError):
    code = "knowledge_base_name_taken"
    default_message = "A knowledge base with this name already exists"


class SourceNotFoundError(DocumentNotFoundError):
    code = "source_not_found"
    default_message = "Source not found"
