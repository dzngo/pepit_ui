import pickle
from pathlib import Path


class PicklePointCache:
    def __init__(self, path: Path):
        self._path = path
        self._cache = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("rb") as handle:
                loaded = pickle.load(handle)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def get(self, key: tuple):
        return self._cache.get(key)

    def set(self, key: tuple, value: tuple) -> None:
        self._cache[key] = value

    def remove_by_algo(self, algo_key: str) -> None:
        keys_to_remove = [key for key in self._cache.keys() if key and key[0] == algo_key]
        for key in keys_to_remove:
            self._cache.pop(key, None)

    def flush(self) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        try:
            with tmp_path.open("wb") as handle:
                pickle.dump(self._cache, handle)
            tmp_path.replace(self._path)
        except Exception:
            return
