from typing import Protocol


class PointExecutorPort(Protocol):
    def compute_points(
        self,
        *,
        work_items: list[tuple[tuple[int, ...], dict[str, object], tuple]],
        algo_key: str,
        function_config: dict[str, dict[str, object]],
        active_dual_series_ids: tuple[str, ...],
    ):
        """Yield tuples:
        (idx_tuple, point_key, tau_value, duals, warning_message, should_cache)
        """
        ...
