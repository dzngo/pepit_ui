import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from algorithm.runtime import compute_point_process


class ProcessPoolPointExecutor:
    def __init__(self, *, max_workers: int | None = None):
        self._max_workers = max_workers

    def compute_points(
        self,
        *,
        work_items: list[tuple[tuple[int, ...], dict[str, object], tuple]],
        algo_key: str,
        function_config: dict[str, dict[str, object]],
        active_dual_series_ids: tuple[str, ...],
    ):
        if not work_items:
            return
        total = len(work_items)
        max_workers = self._max_workers or min(total, max(1, min(8, os.cpu_count() or 1)))
        mp_context = None
        if "fork" in mp.get_all_start_methods():
            mp_context = mp.get_context("fork")

        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
                future_meta = {
                    executor.submit(
                        compute_point_process,
                        algo_key,
                        function_config,
                        algo_params,
                        active_dual_series_ids,
                    ): (idx_tuple, point_key)
                    for idx_tuple, algo_params, point_key in work_items
                }
                for future in as_completed(future_meta):
                    idx_tuple, point_key = future_meta[future]
                    tau_value, duals, warning_message, should_cache = future.result()
                    yield idx_tuple, point_key, tau_value, duals, warning_message, should_cache
        except Exception as pool_exc:
            warning = f"{algo_key}: process parallelism unavailable; falling back to sequential ({pool_exc})"
            for idx_tuple, algo_params, point_key in work_items:
                tau_value, duals, warning_message, should_cache = compute_point_process(
                    algo_key,
                    function_config,
                    algo_params,
                    active_dual_series_ids,
                )
                joined_warning = warning_message
                if joined_warning:
                    joined_warning = f"{warning}\n{joined_warning}"
                else:
                    joined_warning = warning
                yield idx_tuple, point_key, tau_value, duals, joined_warning, should_cache
