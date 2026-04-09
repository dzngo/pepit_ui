from collections.abc import MutableMapping


class SessionGridCache:
    def __init__(self, session_state: MutableMapping[str, object], *, key: str = "tau_grid_cache_nd"):
        self._session_state = session_state
        self._key = key
        self._cache = session_state.setdefault(key, {})
        if not isinstance(self._cache, dict):
            self._cache = {}
            session_state[key] = self._cache

    def get(self, key: tuple):
        return self._cache.get(key)

    def set(self, key: tuple, value: tuple) -> None:
        self._cache[key] = value

    def remove_by_algo(self, algo_key: str) -> None:
        keys_to_remove = [key for key in self._cache.keys() if key and key[0] == algo_key]
        for key in keys_to_remove:
            self._cache.pop(key, None)
