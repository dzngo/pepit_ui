# Interactive Tau Explorer

## Setup
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`

**Or with uv**
- `uv sync`

## Run
- `streamlit run app.py`

## Architecture

Current high-level structure:

```text
app.py

application/               # workflow/use-case services
algorithm/                 # algorithm specs, compiler, runtime
core/                      # compute/config domain logic + ports
infrastructure/            # cache/executor adapters
ui/
  app_shell/               # app-level UI entrypoints
  phases/                  # config/loading/results views
  components/              # reusable UI components
  state/                   # UI session-state helpers
  assets/                  # CSS/JS/HTML frontend assets
```
