"""Async worker (Phase 8).

Celery app + crawl/feature tasks so long-running crawls never block an HTTP
request. Backed by Redis.
"""
