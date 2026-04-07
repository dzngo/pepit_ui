from dataclasses import dataclass

from algorithm.types import HyperparameterSpec


@dataclass(frozen=True)
class ComputeRequest:
    algo_key: str
    function_config: dict[str, dict[str, object]]
    hyperparameter_specs: list[HyperparameterSpec]
    rerun_nan_caches: bool
    selected_dual_series_ids: tuple[str, ...] | None = None
    batch_size: int | None = None
    progress_state_key: str | None = None
