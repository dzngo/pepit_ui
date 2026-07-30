# Interactive PEPit Explorer Usage

This guide explains how to use the Streamlit application. It assumes you already know the relevant PEPit concepts and want to operate the UI effectively.

## Start the App

Install dependencies, then run:

```bash
streamlit run app.py
```

The app opens on the configuration page. Use the `Algorithm` selector at the top to choose the algorithm you want to study.

Built-in algorithms currently available:

- `gradient_descent`
- `subgradient_method`
- `proximal_gradient`
- `accelerated_gradient_convex`
- `epsilon_subgradient`

## Configuration Workflow

The configuration page has three main areas:

- `Algorithm`: view or customize the algorithm code.
- `Hyperparameter config`: define the parameter grid.
- `Functions`: choose the PEPit function objects used by the algorithm.

After configuring these sections, click `Plot` to test the algorithm and start computation.

## Configure Hyperparameters

Use the hyperparameter table to define the grid explored by the app.

Each row has:

- `Name`: identifier used directly by algorithm code, such as `gamma`, `n`, `M`, or `eps`.
- `Label`: display label.
- `Type`: `float` or `int`.
- `Min`: first value in the grid.
- `Max`: last value in the grid.
- `Step`: spacing between values.
- `Default`: fixed value used when another parameter is plotted.

Example:

| Name | Label | Type | Min | Max | Step | Default |
| --- | --- | --- | --- | --- | --- | --- |
| `gamma` | `gamma` | `float` | `0.0` | `1.0` | `0.1` | `0.4` |
| `n` | `n (iterations)` | `int` | `5` | `10` | `1` | `7` |

Use `Use gamma/n quick preset` to reset the table to a simple `gamma` and `n` grid. Use `Reset to algorithm defaults` to restore the selected algorithm's defaults.

## Configure Functions

The `Functions` section defines the PEPit function objects available to the algorithm code.

For each function row:

1. Set `name` to the identifier used in code.
2. Choose a `function type`.
3. Fill any required function parameters.

Example for gradient descent:

| name | function type |
| --- | --- |
| `f` | `SmoothConvexFunction` |

The name is strict: if the row is named `f`, the algorithm code must use `f`. If you rename it to `g`, the algorithm code must use `g`.

Click `Add function` when an algorithm needs multiple functions. For example, proximal gradient uses two functions such as `f1` and `f2`.

## Vary Function Parameters

Some function parameters can be varied by checking `Vary` next to the parameter. When varied, the parameter becomes part of the grid using the name:

```text
function_name.parameter_name
```

Example: if the function row is named `f` and parameter `L` is varied, the app creates a grid parameter named:

```text
f.L
```

You then configure `min`, `max`, `step`, `default`, and `type` for that function parameter just like a normal hyperparameter.

Use fixed function parameters when you want one value for the whole computation. Use varied function parameters when you want to compare tau or dual variables across those values.

## Customize Algorithm Code

The `Algorithm` section shows the selected algorithm body. Click `Customize` to edit it.

The editor expects body-only Python code. Do not include a `def ...` wrapper. Configured function names and hyperparameter names are injected automatically, so you can use names like `f`, `gamma`, and `n` directly.

Example:

```python
xs = f.stationary_point()
xs.set_name("x_*")
fs = f(xs)
x0 = problem.set_initial_point()
problem.set_initial_condition((x0 - xs) ** 2 <= 1)
x = x0

for i in range(n):
    x = x - gamma * f.gradient(x)
    x.set_name(f"x_{i+1}")

problem.set_performance_metric(f(x) - fs)
```

Function parameters are accessed from the function object. For example:

```python
step = 1 / f.L
x = x - step * f.gradient(x)
```

Use `Test` before plotting. The test runs the algorithm with the current default parameter values and reports whether the code is valid for the configured functions and hyperparameters.

Use `Save` to store the edited body as a new custom algorithm. Give it a unique name. The new algorithm appears in the algorithm selector.

Use `Remove customized algorithm` on the configuration page to delete a saved custom algorithm.

## Plot Results

Click `Plot` after configuration.

Before computing the grid, the app checks:

- hyperparameter table validity
- function row validity
- function parameter validity
- algorithm code validity at default parameter values

If validation succeeds, the app enters the computation page.

## Computation Page

The computation page summarizes:

- selected algorithm
- hyperparameter ranges
- algorithm body
- configured functions and function parameters

A progress bar shows grid computation progress.

Click `Interrupt` to cancel the current computation and return to configuration. Use this when the grid is too large, parameter values are wrong, or the algorithm needs editing.

The `Rerun Nan caches` option on the configuration page tells the app to recompute cached points whose tau value is missing or non-finite.

## Results Page

After computation, the app opens the results page.

Use `Configuration details` to review the algorithm body, hyperparameters, and function configuration that produced the results.

Use `Change hyperparameter settings` to return to configuration.

The results workspace displays tau plots and dual-variable controls for the selected parameter axes. When several hyperparameters are present, each plotted parameter uses the other parameters as fixed values. Adjust those fixed values in the results workspace to inspect different slices of the computed grid.

The dual-variable workspace has two modes:

- `Explore plots`: select dual variables and plot their curves.
- `Prepare recompute`: choose dual variables to deactivate, then recompute without them.

Use `Explore plots` when you want to inspect dual-variable behavior. Use `Prepare recompute` when you want to create a new recompute run that removes selected dual variables from the active set.

## Tau Plots

Tau plots show the computed worst-case guarantee over the selected parameter values.

For example, with `gamma` and `n`:

- the tau vs `gamma` plot varies `gamma` and fixes `n`
- the tau vs `n` plot varies `n` and fixes `gamma`

Each plot has its own fixed-parameter controls, so changing the fixed value for one plot does not force the same fixed value in another plot.

If some parameter combinations fail to solve, the app shows gaps in the plot and displays warning messages above the results workspace.

## Dual Variables

Dual variables are shown as color-coded buttons rather than plotted all at once.

Use the ranking metric selector to change how buttons are ranked and colored. Available ranking ideas include:

- non-zero percentage
- standard deviation
- median absolute value
- average absolute value

Dual variables with stronger ranking scores appear more prominent. Variables that are always zero are shown as low-importance entries.

Use `Show all-zero duals` to include or hide dual variables whose plotted values are all zero.

In `Explore plots` mode:

1. Click dual-variable buttons to select the series you want to inspect.
2. Use `Plot selected` to draw the selected dual-variable curves.
3. Use `Select all` or `Deselect all` to quickly toggle all currently visible dual buttons.
4. Click plotted dual cards to select them.
5. Use `Remove selected plotted duals` to remove selected cards from the plotted set.

Selected dual variables appear in dual-variable plot cards for the current parameter slice.

## Curve Prediction

The results workspace has two separate `Curve prediction` toggles:

- one in the `Worst-case guarantee` section for tau plots
- one in `Explore plots` mode for plotted dual-variable curves

Turn on `Curve prediction` to show expression inputs and prediction traces.

Prediction expressions can use:

- `x`, the horizontal-axis value of the current plot
- the plotted parameter name, such as `gamma` or `n`
- other configured hyperparameters, injected at the fixed values selected for the current slice
- varied function parameters, such as `f.L` or `f.mu`, when they are part of the configuration
- fixed scalar function parameters, when they can be injected as numeric values


## Recompute With Selected Dual Variables

The recompute workflow lets you compare the baseline computation with a computation that deactivates selected dual-variable constraints.

Typical workflow:

1. Inspect the dual-variable buttons.
2. Switch from `Explore plots` to `Prepare recompute`.
3. Click dual-variable buttons to add them to `Deactivated dual values`.
4. Click a deactivated value in the list to reactivate it if needed.
5. Use `Deactivate visible duals` to deactivate all currently visible dual buttons.
6. Use `Reactivate all` to clear the deactivated list.
7. Click `Recompute without selected duals`.
8. Compare the new run overlay with the baseline tau and dual-variable plots.

The `Recompute without selected duals` button is enabled only when at least one dual value is deactivated. The recompute run keeps every dual variable active except the ones listed under `Deactivated dual values`.

If you already plotted dual-variable cards, you can also select cards and use `Deactivate selected plotted duals` while in `Prepare recompute` mode.

Each recompute creates a named run such as `Run 1`, `Run 2`, and so on. Runs are displayed as overlays so you can compare how deactivating dual variables affects the guarantee.

You can remove recompute runs from the results workspace when they are no longer useful.

## Save Work

On the results page, click `Save work` to download a binary checkpoint.

The filename follows this pattern:

```text
algorithm-work-YYYYMMDD-HHMMSS.pepit-work.bin
```

The checkpoint stores the current algorithm, configuration, computed state, plot cursors, curve prediction inputs, and recompute runs.

## Load Work

On the configuration page:

1. Click `Load work`.
2. Choose a saved `.pepit-work.bin` file.
3. Click `Load`.

If the file is valid, the app restores the saved work and opens directly on the results page.

If the checkpoint includes a custom algorithm that is not currently loaded, the app restores that custom algorithm too, unless there is a name conflict with an existing algorithm.
