"""Chunking golden tests: every fixture document type through the real ingest pipeline.

The fixture site is treated as one crawl job, in the order the pipeline runs:

1. every HTML page and document is extracted and cleaned (what the crawl engine stores);
2. the job-level boilerplate index is built over all of them (Step 40; the hub page
   first, as the crawl adds it first), so chrome repeated across pages (the cookie
   banner) is stripped;
3. each golden document is stripped, cleaned again (Step 45 prunes emptied sections) and
   chunked with the structure chunker.

Each is chunked with the default settings and with a small budget that forces splitting,
and compared with `tests/fixtures/golden/chunks/<name>.<budget>.txt`. A change anywhere
in extraction, cleaning, boilerplate, the parser, the splitter or the embed-text context
shows up as a reviewable diff. Accept intended changes with `uv run pytest --update-golden`.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_engine.chunking import (
    ChunkDraft,
    ChunkingConfig,
    ChunkInput,
    StructureChunker,
    compose_embed_text,
    tokenizer_for,
)
from rag_engine.config import settings
from rag_engine.extraction import (
    BoilerplateIndex,
    ExtractedDocument,
    ExtractionError,
    ExtractionInput,
    TextCleaner,
    default_registry,
)
from tests.support.fixture_site import SITE_ROOT
from tests.support.golden import GOLDEN_ROOT, assert_golden

_BASE = "https://acme.example"
_TOKENIZER = tokenizer_for(settings.DEFAULT_EMBEDDING_MODEL)
_BUDGETS = {
    "default": ChunkingConfig.from_settings(settings),
    "small": ChunkingConfig(target_tokens=60, max_tokens=160, overlap_tokens=20),
}
_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@dataclass(frozen=True)
class _Doc:
    name: str  # golden file stem
    path: str  # relative to the fixture site root
    mime: str


_DOCUMENTS = [
    _Doc("home", "index.html", "text/html"),
    _Doc("getting-started", "docs/getting-started.html", "text/html"),
    _Doc("configuration", "docs/configuration.html", "text/html"),
    _Doc("api-reference", "docs/api/reference.html", "text/html"),
    _Doc("blog-post", "blog/post-1.html", "text/html"),
    _Doc("latin1", "latin1.html", "text/html"),
    _Doc("widget-spec-pdf", "files/widget-spec.pdf", "application/pdf"),
    _Doc("install-guide-docx", "files/install-guide.docx", _DOCX),
    _Doc("roadmap-pptx", "files/roadmap.pptx", _PPTX),
    _Doc("pricing-xlsx", "files/pricing.xlsx", _XLSX),
]


def _body(path: str) -> bytes:
    raw = (SITE_ROOT / path).read_bytes()
    if path.endswith(".html"):  # the fixture server's per-request templating, made fixed
        raw = raw.replace(b"{{BASE}}", _BASE.encode()).replace(
            b"{{OTHER_HOST}}", b"https://other.acme.example"
        )
    return raw


_MIME_BY_SUFFIX = {
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".docx": _DOCX,
    ".pptx": _PPTX,
    ".xlsx": _XLSX,
}


@dataclass(frozen=True)
class _Corpus:
    documents: dict[str, ExtractedDocument]  # site path → extraction
    cleaned: dict[str, str]  # site path → stored (cleaned) Markdown
    index: BoilerplateIndex


async def _build_corpus() -> _Corpus:
    cleaner = TextCleaner(config=settings)
    registry = default_registry(settings)
    paths = sorted(
        str(p.relative_to(SITE_ROOT))
        for p in SITE_ROOT.rglob("*")
        if p.suffix in _MIME_BY_SUFFIX and p.is_file()
    )
    paths.remove("index.html")
    documents: dict[str, ExtractedDocument] = {}
    cleaned: dict[str, str] = {}
    index = BoilerplateIndex(config=settings)
    for path in ["index.html", *paths]:  # the hub page first, like the crawl
        mime = _MIME_BY_SUFFIX[f".{path.rsplit('.', 1)[-1]}"]
        try:
            document = await registry.extract(
                ExtractionInput(url=f"{_BASE}/{path}", mime_type=mime, body=_body(path))
            )
        except ExtractionError:
            continue  # pages the crawl records as skipped/failed (scanned PDF, empty shell)
        documents[path] = document
        cleaned[path] = cleaner.clean(document.markdown).markdown
        index.add(path, cleaned[path])
    return _Corpus(documents, cleaned, index)


@pytest.fixture(scope="module")
def corpus() -> _Corpus:
    return asyncio.run(_build_corpus())


def _chunk(
    doc: _Doc, config: ChunkingConfig, corpus: _Corpus
) -> tuple[ChunkInput, list[ChunkDraft]]:
    document = corpus.documents[doc.path]
    stripped = corpus.index.strip(doc.path, corpus.cleaned[doc.path]).markdown
    markdown = TextCleaner(config=settings).clean(stripped).markdown
    source = ChunkInput(
        tenant_id="ten_" + "1" * 32,
        kb_id="kb_" + "1" * 32,
        source_id="src_" + "1" * 32,
        version="job_" + "1" * 32,
        page_id="page_" + "1" * 32,
        url=f"{_BASE}/{doc.path}",
        markdown=markdown,
        content_kind=document.content_kind,
        title=document.title,
        lang=document.lang,
    )
    return source, StructureChunker(_TOKENIZER, config).chunk(source)


def _render(source: ChunkInput, drafts: list[ChunkDraft], config: ChunkingConfig) -> str:
    lines = [
        f"# {source.url}",
        f"# title: {source.title!r}",
        f"# budget: target={config.target_tokens} max={config.max_tokens} "
        f"overlap={config.overlap_tokens} context={config.context_tokens}",
        f"# chunks: {len(drafts)}",
        "",
    ]
    for draft in drafts:
        context = draft.embed_text.removesuffix(draft.text).removesuffix("\n\n")
        path = " > ".join(draft.heading_path) or "(no headings)"
        location = draft.page_number if draft.page_number is not None else "-"
        lines += [
            f"=== chunk {draft.ordinal} | {path} | location {location} | "
            f"{draft.token_count} tokens ===",
            "--- context ---",
            context or "(none)",
            "--- text ---",
            draft.text,
            "",
        ]
    return "\n".join(lines)


# ---- #
# Tests
# ---- #


class TestChunkingGolden_S60:
    """S60 — Chunking golden tests: snapshots of every fixture document type, two budgets."""

    @pytest.mark.parametrize("budget", sorted(_BUDGETS))
    @pytest.mark.parametrize("doc", _DOCUMENTS, ids=lambda d: d.name)
    def test_chunks_match_the_golden_file(
        self, doc: _Doc, budget: str, corpus: _Corpus, request: pytest.FixtureRequest
    ) -> None:
        config = _BUDGETS[budget]

        source, drafts = _chunk(doc, config, corpus)

        # the contract holds before the snapshot is even compared
        assert drafts, f"{doc.name} produced no chunks"
        assert [d.ordinal for d in drafts] == list(range(len(drafts)))
        for draft in drafts:
            assert 0 < draft.token_count <= config.max_tokens
            assert draft.embed_text == compose_embed_text(
                draft.embed_text.removesuffix(draft.text).removesuffix("\n\n"), draft.text
            )
            assert "<!--" not in draft.text
        assert_golden(
            GOLDEN_ROOT / "chunks" / f"{doc.name}.{budget}.txt",
            _render(source, drafts, config),
            update=request.config.getoption("--update-golden"),
        )

    def test_chunking_is_deterministic(self, corpus: _Corpus) -> None:
        doc = _DOCUMENTS[1]

        first = _chunk(doc, _BUDGETS["small"], corpus)
        second = _chunk(doc, _BUDGETS["small"], corpus)

        assert first[1] == second[1]

    def test_site_chrome_never_reaches_a_chunk(self, corpus: _Corpus) -> None:
        assert corpus.index.is_active()  # enough pages for the job-level strip

        texts = [d.text for doc in _DOCUMENTS for d in _chunk(doc, _BUDGETS["default"], corpus)[1]]

        assert not any("We use cookies" in text for text in texts)

    def test_a_changed_output_fails_with_a_diff(self, tmp_path: Path) -> None:
        golden = tmp_path / "doc.txt"
        assert_golden(golden, "line one\nline two\n", update=True)

        assert_golden(golden, "line one\nline two\n", update=False)  # unchanged: passes
        with pytest.raises(AssertionError, match=r"(?s)-line two.*\+line 2"):
            assert_golden(golden, "line one\nline 2\n", update=False)
        with pytest.raises(AssertionError, match="missing"):
            assert_golden(tmp_path / "absent.txt", "x", update=False)

    def test_every_fixture_document_type_is_covered(self) -> None:
        mimes = {doc.mime for doc in _DOCUMENTS}

        assert {"text/html", "application/pdf", _DOCX, _PPTX, _XLSX} <= mimes
        names = [doc.name for doc in _DOCUMENTS]
        assert len(names) == len(set(names))
