"""CLI entry point — ``uv run python -m rag_engine.run --url <URL> [--full-site]``.

Phase 6 wires this to the ETL pipeline (link → crawler → Mongo). For now it
parses arguments and reports that ingestion is not yet implemented.
"""

import argparse

from rag_engine.config import logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_engine.run",
        description="Ingest a website into the RAG Engine.",
    )
    parser.add_argument("--url", required=True, help="Root URL to crawl and ingest.")
    parser.add_argument(
        "--full-site",
        action="store_true",
        help="Crawl the whole site (sitemap/BFS) rather than a single page.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger.info(
        "requested ingestion of {} (full_site={}) — not yet implemented (Phase 6)",
        args.url,
        args.full_site,
    )


if __name__ == "__main__":
    main()
