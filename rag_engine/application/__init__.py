"""Application layer — crawlers and preprocessing pipelines.

Swappable concerns follow the house dispatcher seam: ``base.py`` (ABC) +
``dispatcher.py`` (Factory → Dispatcher switching on a ``StrEnum``) + concrete
implementations + ``exception.py``.
"""
