from pymongo import ASCENDING, IndexModel

from rag_engine.domain.base import MongoDocument
from rag_engine.domain.types import HttpUrlStr, JobId, PageId, Sha256Hex

# claims only matter while their job runs; this is a safety net for jobs that never finalize
CLAIM_TTL_SECONDS = 30 * 24 * 60 * 60


class ContentHashClaim(MongoDocument):
    """First page of a crawl job with a given content hash (Step 46).

    The unique `(job_id, content_hash)` index makes "first page wins" atomic across
    concurrent fetches and survives a crash, so a resumed job keeps the same originals.
    """

    collection_name = "content_hash_claims"
    id_prefix = "chc"
    indexes = (
        IndexModel([("job_id", ASCENDING), ("content_hash", ASCENDING)], unique=True,
                   name="job_content_unique"),
        IndexModel([("created_at", ASCENDING)], expireAfterSeconds=CLAIM_TTL_SECONDS,
                   name="claim_ttl"),
    )  # fmt: skip

    job_id: JobId
    content_hash: Sha256Hex
    page_id: PageId
    url: HttpUrlStr
