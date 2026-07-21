"""Crawler framework (Phases 3–4).

``BaseCrawler`` + ``CrawlerDispatcher`` (regex on domain → concrete crawler,
fallback → generic article crawler), plus the concrete crawlers
(generic-article, full-site, Selenium, GitHub).
"""
