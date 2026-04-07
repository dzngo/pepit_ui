"""UI view aggregator.

Phase-specific entrypoints are split across dedicated modules.
"""

from ui.app_shell.shared import init_session_state, reset_for_algorithm_change
from ui.phases.config_view import render_config_phase
from ui.phases.loading_view import render_loading_phase
from ui.phases.results_view import render_results_phase

__all__ = [
    "init_session_state",
    "reset_for_algorithm_change",
    "render_config_phase",
    "render_loading_phase",
    "render_results_phase",
]
