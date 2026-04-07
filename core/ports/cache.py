from collections.abc import MutableMapping
from typing import Protocol


class PointCachePort(Protocol):
    def get(self, key: tuple):
        ...

    def set(self, key: tuple, value: tuple) -> None:
        ...

    def remove_by_algo(self, algo_key: str) -> None:
        ...

    def flush(self) -> None:
        ...


class GridCachePort(Protocol):
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
