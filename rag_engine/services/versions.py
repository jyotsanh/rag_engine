"""Version swap semantics (Step 73): search serves exactly each source's active version.

A crawl writes its chunks and vectors under its own `version` (the job id). Nothing it
writes is visible to search until finalization (Step 78) flips `Source.active_version`
in one atomic Mongo update (`SourceRepository.activate_version`). Because every search
filters on `version ∈ active versions of the KB` (`search_scope`), a re-crawl in progress
is invisible, the old version keeps serving until the flip, and the flip switches every
search at once. The old version's points stay in Qdrant, unserved, until cleanup.

**Safety rule**: an empty `versions` tuple means "no constraint" in `VectorFilter`, so a
KB with no active version at all (only first crawls in progress) must not be searched
with a version-less filter, which would serve half-ingested points. `search_scope`
returns `None` instead, and the caller answers with no results.
"""

from rag_engine.repositories import SourceRepository
from rag_engine.vectorstores import VectorFilter


async def search_scope(
    sources: SourceRepository,
    tenant_id: str,
    kb_id: str,
    *,
    source_ids: tuple[str, ...] = (),
    url_prefix: str | None = None,
) -> VectorFilter | None:
    """The vector filter for searching a KB (optionally some of its sources), or `None`
    when nothing is servable yet (no source has an active version)."""
    active = await sources.active_versions(tenant_id, kb_id, source_ids=source_ids)
    if not active:
        return None
    return VectorFilter(
        tenant_id,
        kb_ids=(kb_id,),
        source_ids=tuple(sorted(active)),
        versions=tuple(sorted(set(active.values()))),
        url_prefix=url_prefix,
    )
