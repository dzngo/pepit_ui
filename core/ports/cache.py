from collections.abc import MutableMapping
from typing import Protocol


class PointCachePort(Protocol):
    """Persistent point-level cache.

    Stores/retrieves individual parameter-point evaluations and survives across
    app reruns/sessions when backed by disk.
    """

    def get(self, key: tuple):
        ...

    def set(self, key: tuple, value: tuple) -> None:
        ...

    def remove_by_algo(self, algo_key: str) -> None:
        ...

    def flush(self) -> None:
        ...


class GridCachePort(Protocol):
    """Session-scoped full-grid cache.

    Stores/retrieves fully assembled N-D compute results for the current app
    session (typically backed by Streamlit session state).
    """

    def get(self, key: tuple):
        ...

    def set(self, key: tuple, value: tuple) -> None:
        ...

    def remove_by_algo(self, algo_key: str) -> None:
        ...


class ProgressStorePort(Protocol):
    def get(self, key: str, default=None):
        ...

    def __setitem__(self, key: str, value) -> None:
        ...


ProgressStore = MutableMapping[str, object]
