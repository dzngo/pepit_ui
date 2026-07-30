# Interactive PEPit Explorer

Interactive PEPit Explorer is a Streamlit application for exploring numerical worst-case guarantees produced by [PEPit](http://pepit.readthedocs.io/). It lets a user configure an algorithm, compute and inspect tau and dual-variable series, and recompute them with some deactivated dual variables.

This README is written for future maintainers. For application usage instructions, see [USAGE.md](USAGE.md).

## Requirements

- Python 3.12 or newer
- `uv` or `pip`

Dependencies are declared in [pyproject.toml](pyproject.toml). The PEPit dependency is pinned to a Git revision because this app depends on PEPit's dual-variable activation/deactivation support, which was not available from the released PyPI package when the recompute feature was developed:

```toml
[tool.uv.sources]
pepit = { git = "https://github.com/dzngo/PEPit.git", rev = "f851d4fcb9f9a69dbf0bb53873240cfe43b3b35b" }
```

This commit comes from a fork used as a temporary packaging workaround. Installing the upstream PEPit commit directly failed because the unreleased package still contained `{{VERSION_PLACEHOLDER}}` in its packaging metadata, which caused `uv` and `pip` builds to fail with an invalid-version error. Using a submodule was also avoided because it complicated imports and deployment, especially for Streamlit Cloud. The forked commit sets a concrete development version and adjusts packaging metadata so `uv` and `pip` can install it from Git.

Once the required PEPit feature is released on PyPI, this dependency should be revisited and ideally replaced with a normal version constraint. Contact Baptiste Goujaud or Adrien Taylor for the status of the upstream PEPit release and whether the pinned fork is still needed.

## Setup

With `uv`:

```bash
uv sync
```

With `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The Streamlit entrypoint is [app.py](app.py). It initializes session state, renders the algorithm selector, and dispatches to one of three UI phases: configuration, loading, or results.

## Source Layout

```text
app.py
algorithm/          # algorithm templates, custom algorithm persistence, compiler, runtime
core/               # Streamlit-independent compute/config logic and ports
infrastructure/     # cache and point-execution adapters
service/            # workflow services between UI and core layers
ui/                 # Streamlit views, reusable components, state helpers, browser assets
draft_scripts/      # exploratory scripts and UI experiments
```

Main responsibilities:

- [algorithm/algorithm_templates.py](algorithm/algorithm_templates.py) defines built-in algorithm bodies and their default hyperparameters/functions.
- [algorithm/algorithm_custom.py](algorithm/algorithm_custom.py) loads, registers, saves, and removes custom algorithms.
- [algorithm/algorithm_compiler.py](algorithm/algorithm_compiler.py) compiles body-only algorithm code and injects configured names into the execution namespace.
- [algorithm/runtime.py](algorithm/runtime.py) builds a PEPit problem, declares configured functions, runs the algorithm, solves the PEP, and extracts dual values.
- [core/compute/engine.py](core/compute/engine.py) computes or resumes N-dimensional parameter grids and owns cache-key normalization.
- [core/compute/series.py](core/compute/series.py) converts N-dimensional tau/dual arrays into plot series and dual-ranking data.
- [service/config_service.py](service/config_service.py), [service/loading_service.py](service/loading_service.py), [service/results_service.py](service/results_service.py), and [service/workspace_io_service.py](service/workspace_io_service.py) orchestrate phase transitions, recompute events, and checkpoint import/export.
- [ui/phases](ui/phases) contains the three Streamlit phase views.
- [ui/components](ui/components) contains the algorithm editor, configuration panels, and results workspace bridge.
- [ui/assets/results_workspace](ui/assets/results_workspace) contains the no-build JavaScript/CSS workspace used for interactive results.

## Application Flow

The app moves through three phases stored in `st.session_state["ui_phase"]`.

1. `config`

   Rendered by [ui/phases/config_view.py](ui/phases/config_view.py). The user edits hyperparameters, functions, and optional custom algorithm code. Clicking `Plot` validates the configuration and performs a smoke test before creating `pending_settings`.

2. `loading`

   Rendered by [ui/phases/loading_view.py](ui/phases/loading_view.py). The app computes the grid in small batches through a Streamlit fragment so the progress UI and `Interrupt` button stay responsive. On success, `pending_settings` becomes `active_settings`.

3. `results`

   Rendered by [ui/phases/results_view.py](ui/phases/results_view.py). The app builds tau series, dual-variable series, dual-ranking sections, recompute runs, and checkpoint export data. Results events from the browser component update Streamlit session state and trigger reruns.

## Algorithms

Built-in algorithms are declared in [algorithm/algorithm_templates.py](algorithm/algorithm_templates.py). Each `AlgorithmSpec` contains:

- `name`: algorithm key shown in the selector
- `algo`: compiled callable
- `default_hyperparameters`: rows used to initialize the hyperparameter editor
- `default_function_rows`: function rows used to initialize the function panel

Custom algorithms are saved through the UI into `custom_algorithms.json`. A custom algorithm stores its body-only code, source/base algorithm, default hyperparameter rows, default function rows, and creation timestamp.

If a custom algorithm becomes part of the maintained application rather than local user state, promote it into [algorithm/algorithm_templates.py](algorithm/algorithm_templates.py) instead of relying on `custom_algorithms.json`.

## Function and Parameter Model

The function registry in [algorithm/function_registry.py](algorithm/function_registry.py) introspects `PEPit.functions.__all__` and builds UI metadata from each function class constructor.

Function row names are canonical runtime identifiers. For example, a row named `f` is exposed to algorithm code as `f`. Function parameters can be fixed values or varied values. Varied function parameters become hyperparameters named with dotted identifiers such as `f.L` or `f.mu`; at runtime, [algorithm/runtime.py](algorithm/runtime.py) applies those values as constructor overrides for the matching function object.

Algorithm hyperparameters and function names must not conflict. The runtime also excludes internal names such as `problem`, `funcs`, `params`, `PEP`, `Point`, `Function`, `np`, and `sqrt`.

## Compute and Cache Layers

Grid computation is routed through [infrastructure/compute_runner.py](infrastructure/compute_runner.py), which wires the core engine to Streamlit session state and persistent point caching.

There are two cache levels:

- Session grid cache: full-grid results stored in Streamlit session state under `tau_grid_cache_nd`.
- Persistent point cache: point-level results stored in `.tau_point_cache.pkl`.

Cache keys normalize the algorithm key, hyperparameter specs, function configuration, concrete parameter assignment, and selected dual-variable series. This allows baseline and recompute runs to share point results when the effective inputs match.

The `Rerun Nan caches` option recomputes cached points whose tau value is missing or non-finite.

## Results and Recompute

The results UI is rendered by a custom browser component assembled in [ui/components/results_panel.py](ui/components/results_panel.py) from files in [ui/assets/results_workspace](ui/assets/results_workspace).

The component sends structured events back to Streamlit:

- cursor changes for plot slices and local fixed-parameter values
- ranking metric changes
- recompute requests
- recompute-run removal

[service/results_service.py](service/results_service.py) validates and applies those events, then stores recompute runs under algorithm-scoped session-state keys such as `recompute_runs_<algo_key>`.

Recompute runs store the active dual-variable series, the deactivated series, display labels, visibility, and run id. They are plotted as overlays against the baseline tau and dual-variable data.

## Checkpoints

Work checkpoints are binary files generated from [service/workspace_io_service.py](service/workspace_io_service.py). Export is available from the results phase through the `Save work` button.

A checkpoint contains:

- checkpoint version
- save timestamp
- selected algorithm key
- active settings
- custom algorithm payload, when needed
- UI state such as ranking metric, cursors, pattern inputs, recompute runs, and event ids
- session-grid cache subset for the selected algorithm
- stored hyperparameter/function rows

Loading a checkpoint validates the version and algorithm payload, restores session state, and returns directly to the results phase.

## Generated and Local Artifacts

The following files are generated or user-local by default:

- `.tau_point_cache.pkl`: persistent compute cache; do not source-control.
- `custom_algorithms.json`: local saved custom algorithms; source-control only when the saved algorithms are intentionally shared.
- `*.pepit-work.bin`: downloaded work checkpoints; do not source-control by default.
- `.DS_Store`: macOS metadata; do not source-control.

## Extension Points

- Add a built-in algorithm by adding a body to `BASE_ALGORITHM_BODIES` and an `AlgorithmSpec` to `BASE_ALGORITHMS` in [algorithm/algorithm_templates.py](algorithm/algorithm_templates.py).
- Change algorithm-code execution rules in [algorithm/algorithm_compiler.py](algorithm/algorithm_compiler.py).
- Change PEPit function discovery or parameter parsing in [algorithm/function_registry.py](algorithm/function_registry.py) and [ui/components/config/functions_panel.py](ui/components/config/functions_panel.py).
- Change compute/cache behavior in [core/compute/engine.py](core/compute/engine.py), [infrastructure/cache.py](infrastructure/cache.py), and [infrastructure/compute_runner.py](infrastructure/compute_runner.py).
- Change result-event handling in [service/results_service.py](service/results_service.py).
- Change the interactive result workspace in [ui/assets/results_workspace](ui/assets/results_workspace).

Keep Streamlit-specific behavior in `ui/`, `service/`, or `infrastructure/` where possible. The `core/` package is the best place for logic that can stay independent from Streamlit reruns and session state.
