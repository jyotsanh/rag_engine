from abc import ABC, abstractmethod
from typing import Any


class BaseDatabase(ABC):
    """Contract every raw-document warehouse backend implements.

    Keeps the app backend-agnostic: the lifespan and (later) the ODM/repository
    layer depend only on this interface, so swapping MongoDB for another store
    means adding one implementation + one ``DatabaseFactory`` branch — no call
    sites change.
    """

    @abstractmethod
    def connect(self) -> None:
        """Open the connection to the backend. Lazy drivers may defer real I/O."""

    @abstractmethod
    def ping(self) -> bool:
        """Verify the backend is reachable. Raise ``DatabaseConnectionError`` if not."""

    @abstractmethod
    def close(self) -> None:
        """Release the connection. Safe to call if never connected."""

    @abstractmethod
    def get_database(self) -> Any:
        """Return the native database handle for the ODM / repository layer (Phase 2)."""
