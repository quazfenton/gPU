# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

`gPU` is a single-file Python CLI (`runna.py`) for automating Kaggle notebook
workflows and deploying notebooks to serverless platforms: Google Cloud
Functions, AWS Lambda, Modal.com, and a local HTTP server.

- **Language:** Python 3.7+ (README states 3.8+; `doctor.py` enforces 3.7+)
- **No build system / packaging:** plain `requirements.txt`, no `setup.py`,
  `pyproject.toml`, or `Makefile`. Install deps with
  `pip install -r requirements.txt`.
- **No CI:** there are no `.github/workflows` or other CI configs.

## Common Commands

### Run the CLI

All commands go through `runna.py` (argparse subcommands). Examples:

```bash
python3 runna.py --help
python3 runna.py <command> --help
python3 runna.py doctor          # validate environment (Kaggle CLI, gcloud, credentials)
python3 runna.py --validate      # same validation, invoked via top-level flag
```

Full subcommand list (defined in `runna.py:main()` around line 2730):

| Command | Purpose |
|---------|---------|
| `list [--user] [--search]` | List Kaggle kernels |
| `pull <kernel> [--dest]` | Download a kernel |
| `push <path>` | Upload a notebook |
| `create [input]` | Create a new kernel |
| `run [input] [--deploy] ...` | Run a kernel; optional GCP deploy |
| `deploy <path> ...` | Deploy a notebook dir/file to GCP Functions |
| `package-aws <path>` | Package notebook as Lambda zip (no deploy) |
| `deploy-aws <path> ...` | Deploy to AWS Lambda (needs `--role-arn`) |
| `deploy-modal <path> [--gpu T4\|A10G\|A100] ...` | Deploy to Modal.com |
| `serve-local <path> [--port] [--run]` | Package + optionally run local server |
| `app-list` / `app-show <name>` | Browse the app library (`apps/library.json`) |
| `app-deploy <name> [--gpu] [--secrets]` | Deploy a pre-built app |
| `app-add <name> --file ...` | Add an app to the library |
| `app-update <name> --file` / `app-delete <name>` | Manage library |
| `send <endpoint> <text> [--field]` | Send text to a deployed endpoint |
| `chat <endpoint> [--field]` | Interactive chat with an endpoint |
| `endpoints` | List saved endpoints |
| `call <name_or_url> [--method] [--json] [--json-file]` | Call an endpoint with JSON |
| `batch <file> [--operation download\|deploy\|run] [--parallel]` | Batch process notebooks |
| `preprocess <path> [--clean-metadata] [--remove-outputs] [--scan-security] [--optimize-imports] [--output]` | Clean/secure notebooks |
| `doctor` | Validate environment (Kaggle CLI, gcloud, credentials) — same as `--validate` |

### Tests

There is **no formal test framework** (no pytest/unittest). Verification is done
with shell integration scripts and environment checks:

```bash
bash test_app_library.sh    # exercises app-list/show/add/update/delete
bash test_modal.sh           # exercises deploy-modal (requires Modal auth)
bash test_send.sh            # exercises send/chat + app library CRUD (mostly dry-run)
python3 doctor.py            # standalone env checker (Python, Kaggle pkg, creds, API)
```

> Note: `test_modal.sh` requires `modal token new` auth; `test_app_library.sh`
> and `test_send.sh` add/delete entries in `apps/library.json`.

### Lint / Typecheck

There are **no configured linters or type checkers** (no ruff, flake8, pylint,
mypy, or black configs). When editing Python, keep style consistent with
`runna.py`: 4-space indent, `logging` module for output, `except Exception`
(bare `except:` clauses have been migrated out — do not reintroduce them).

## Code Conventions

- **Single-file CLI:** `runna.py` holds all logic. Each subcommand is a
  `cmd_<name>(args)` function wired to its argparse subparser via
  `parser.set_defaults(func=cmd_<name>)`.
- **Logging:** use the module-level `logger` (`logging.getLogger(__name__)`);
  do not `print()` for status/error messages in `runna.py`.
- **Error handling:** `try/except` with `except Exception as e`; log and
  re-raise or return gracefully. Avoid bare `except:`.
- **Type hints:** used sparingly (e.g. `str | None`); follow existing style.
- **Notebook conversion:** Jupyter magic commands are stripped by
  `clean_jupyter_magic_commands()` — preserve this when touching deploy paths.

## Key Files & Layout

```
runna.py              # main CLI (~3100 lines)
app_library.py        # app library manager (backed by apps/library.json)
doctor.py             # standalone environment validator
deploy_model.py       # model download/load/predict (Kaggle + HuggingFace)
modal_deploy.py       # Modal.com deployment template
deploy.sh             # gcloud functions deploy wrapper
requirements.txt      # Python dependencies
apps/                 # app library: .py apps + library.json index
examples/             # sample notebook + Modal example
deploy/               # generated deploy artifacts (main.py, deploy_model.py)
kernel-metadata.json  # Kaggle kernel metadata
.kaggle_state/        # runtime endpoint registry (endpoints.json), not committed
```

## State & Persistence

- **Endpoint registry:** `.kaggle_state/endpoints.json` (created at runtime by
  `runna.py`; not in `.gitignore` but is a runtime artifact — do not commit).
  Endpoints are registered by passing `--save-name <name>` to `run`/`deploy`/
  `deploy-aws`/`deploy-modal` (`register_endpoint()` in `runna.py`); the saved
  name is then reused by `endpoints`, `send`, `chat`, and `call`.
- **App library:** `apps/library.json` + `apps/<name>.py` (committed; managed
  via `app-add`/`app-update`/`app-delete`).
- **Gitignored:** `.env*`, `env`, `venv`, `.kaggle`.

## Deployment Workflows

- **GCP Functions:** `deploy.sh` (calls `gcloud functions deploy`) or
  `runna.py deploy` / `runna.py run --deploy` (needs `--gcp-project`).
- **AWS Lambda:** `runna.py deploy-aws` (requires `--role-arn`); zip-only via
  `runna.py package-aws`.
- **Modal.com:** `runna.py deploy-modal` (GPU: T4/A10G/A100) or
  `runna.py app-deploy <name>` from the library.
- **Local:** `runna.py serve-local <path> --run --port 8080`.

## Credentials

Kaggle credentials are resolved in order (see `get_auth()` in `runna.py`):
1. `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars
2. `~/.kaggle/kaggle.json`
3. `.env` file (via python-dotenv)
4. `kagglehub` SDK

Cloud providers: `gcloud auth login` + `gcloud config set project <ID>` (GCP),
`aws configure` (AWS), `modal token new` (Modal).

Provider config env-var fallbacks (used when the matching flag is omitted):
- **GCP:** `--gcp-project` → `GOOGLE_CLOUD_PROJECT` / `PROJECT_ID`; `--region` →
  `REGION` (default `us-central1`).
- **AWS:** `--role-arn` → `AWS_LAMBDA_ROLE_ARN`; `--region` → `AWS_REGION`
  (default `us-east-1`).
- **`deploy.sh`** (standalone GCP wrapper): reads `FUNCTION_NAME`,
  `GOOGLE_CLOUD_PROJECT`/`PROJECT_ID`, `REGION`, and model env vars
  (`MODEL_SOURCE`, `MODEL_PATH`, `KAGGLE_DATASET`, `KAGGLE_COMPETITION`,
  `KAGGLE_NOTEBOOK`, `HF_REPO_ID`, `HF_MODEL_FILE`).
