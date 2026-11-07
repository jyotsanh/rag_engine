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


class PageNotFoundError(DocumentNotFoundError):
    code = "page_not_found"
    default_message = "Page not found"


class JobNotFoundError(DocumentNotFoundError):
    code = "job_not_found"
    default_message = "Crawl job not found"


class InvalidJobTransitionError(ConflictError):
    code = "invalid_job_transition"
    default_message = "The crawl job cannot change to that status from its current one"


class JobRunSupersededError(ConflictError):
    """The job is no longer held by this worker run (requeued, finished or cancelled)."""

    code = "job_run_superseded"
    default_message = "The crawl job is no longer held by this run"
