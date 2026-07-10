# AGENTS.md

Agent-oriented reference for the gPU repository — Kaggle notebook automation and serverless deployment CLI.

## Project Overview

A Python CLI tool (`runna.py`) that automates Kaggle notebook workflows and deploys them as serverless APIs to Google Cloud Functions, AWS Lambda, or Modal.com. Includes an app library system for reusable Modal templates, an endpoint registry, and notebook preprocessing.

## Quick Setup

```bash
pip install -r requirements.txt
python3 runna.py doctor  # verify environment
```

### Authentication (Kaggle)

Four sources checked in order:
1. `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars
2. `~/.kaggle/kaggle.json`
3. `.env` file in project root (requires `python-dotenv`)
4. `kagglehub` SDK (fallback)

Cloud providers: `gcloud auth login`, `aws configure`, `modal token new`.

## CLI Commands (`python3 runna.py`)

### Kernel Operations

| Command | Args | Description |
|---------|------|-------------|
| `list` | `[--user U] [--search S]` | List Kaggle kernels |
| `pull` | `<kernel_ref_or_url> [--dest DIR]` | Download kernel with metadata |
| `push` | `<path>` | Upload kernel (auto-generates metadata) |
| `create` | `[<input>]` | Create kernel from file/URL or blank |
| `run` | `[<input>] [--dest DIR] [--deploy] [--gcp-project ...]` | Push + wait for completion; optional deploy |

### Deployment

| Command | Args | Description |
|---------|------|-------------|
| `deploy` | `<path> [--gcp-project ...] [--save-name NAME]` | Deploy notebook dir to GCP Functions |
| `deploy-aws` | `<path> [--role-arn ARN] [--save-name NAME]` | Deploy to AWS Lambda |
| `deploy-modal` | `<path> [--gpu TYPE] [--secrets ...] [--timeout SEC] [--save-name NAME]` | Deploy to Modal.com |
| `package-aws` | `<path>` | Package as AWS Lambda zip (no deploy) |
| `serve-local` | `<path> [--port PORT] [--run]` | Package for local HTTP server |

### App Library

| Command | Args | Description |
|---------|------|-------------|
| `app-list` | | List all apps in library |
| `app-show` | `<name>` | Show app source code |
| `app-deploy` | `<name> [--gpu TYPE] [--secrets ...] [--timeout SEC]` | Deploy app from library |
| `app-add` | `<name> --file PATH [--description ...] [--tags ...] [--gpu TYPE]` | Add app to library |
| `app-update` | `<name> --file PATH` | Update app code |
| `app-delete` | `<name>` | Delete app from library |

### Endpoint & Chat

| Command | Args | Description |
|---------|------|-------------|
| `endpoints` | | List saved endpoints |
| `call` | `<name_or_url> [--method M] [--json ...] [--json-file ...]` | HTTP call to endpoint |
| `send` | `<endpoint> <text> [--field NAME]` | Send text to endpoint |
| `chat` | `<endpoint> [--field NAME]` | Interactive chat session |

### Utility

| Command | Args | Description |
|---------|------|-------------|
| `doctor` | | Validate environment (Kaggle CLI, gcloud, credentials) |
| `batch` | `<input_file> --operation {download,deploy,run} [--parallel] [--output-dir DIR]` | Batch process notebooks |
| `preprocess` | `<input_path> [--clean-metadata] [--remove-outputs] [--scan-security] [--optimize-imports] [--output PATH]` | Preprocess notebooks |

## Key Workflows

### 1. Kaggle Kernel → Serverless API

```bash
python3 runna.py pull username/kernel-name
python3 runna.py deploy username_kernel-name/ --gcp-project MY_PROJECT --save-name my-api
```

Or end-to-end:
```bash
python3 runna.py run username/kernel-name --deploy --gcp-project MY_PROJECT --save-name my-api
```

### 2. App Library: Template → Deploy → Interact

```bash
python3 runna.py app-list
python3 runna.py app-deploy llm-chat
python3 runna.py chat llm-chat
```

### 3. Batch Notebook Processing

Create `notebooks.txt` (one ref per line):
```
username/kernel1
username/kernel2
```
```bash
python3 runna.py batch notebooks.txt --operation download --output-dir ./downloads
```

### 4. Endpoint Registry: Deploy → Reuse

```bash
python3 runna.py deploy-modal ./nb --save-name prod-model
python3 runna.py send prod-model "Hello"
python3 runna.py call prod-model --json '{"features": [1,2,3]}'
```

### 5. Local Development Loop

```bash
python3 runna.py serve-local ./notebook-dir --run --port 8080
curl -X POST http://localhost:8080 -H "Content-Type: application/json" -d '{"features": [1,2,3]}'
```

## Built-in App Templates

| App | GPU | Use Case |
|-----|-----|----------|
| `image-classifier` | T4 | ResNet50 image classification |
| `text-generator` | T4 | GPT-2 text generation |
| `llm-chat` | A10G | Interactive LLM chat |
| `web-scraper` | None | BeautifulSoup web scraping |
| `batch-processor` | None | Pandas batch data processing |
| `scheduled-task` | None | Hourly cron task example |

## Key Source Files

| File | Purpose |
|------|---------|
| `runna.py` | Main CLI — all commands, kernels API, deployment, packaging |
| `app_library.py` | App library CRUD (load/save/list/add/update/delete) |
| `doctor.py` | Environment diagnostic checks |
| `deploy_model.py` | Model loading, Kaggle/HF interactions, preprocessing |
| `modal_deploy.py` | Standalone Modal app template |
| `deploy.sh` | Shell script for `gcloud functions deploy` |
| `apps/library.json` | App library index |
| `apps/*.py` | App template source files |

## Important Internals

- **Endpoint registry** stored in `.kaggle_state/endpoints.json` — used by `call`, `send`, `chat` to resolve names to URLs.
- **Notebook conversion** uses `nbconvert` with fallback to manual cell extraction; `clean_jupyter_magic_commands()` strips `%%`, `%`, `get_ipython()`.
- **Dependency detection** (`_detect_model_requirements()`) scans `deploy_model.py` for imports (torch, sklearn, numpy, pandas, etc.) and adds them to generated `requirements.txt`.
- **Kernel metadata** (`kernel-metadata.json`) is auto-generated on push with unique slug (timestamp-based).
- **GPU types for Modal**: `T4` ($0.60/hr inference), `A10G` ($1.10/hr balanced), `A100` ($4.00/hr training).

## Discovered Workflows

### 6. Remote Notebook URL → Deploy

`runna.py` accepts raw notebook URLs and auto-converts them before pulling:

```bash
# GitHub gist, GitLab, Bitbucket, or direct .ipynb URLs
python3 runna.py run https://github.com/user/repo/blob/main/notebook.ipynb --deploy --gcp-project MY_PROJECT
```

Internal pipeline: `_normalize_notebook_url()` → `_convert_github_gist_to_raw()` / `_convert_gitlab_to_raw()` / `_convert_bitbucket_to_raw()` / `_maybe_github_blob_to_raw()` → `download_remote_notebook()` → `_push_and_run_from_source()`.

- **GitHub blob URLs** are converted to raw `raw.githubusercontent.com` URLs.
- **GitHub gists** are resolved to the raw gist file.
- **GitLab** `/-/blob/` paths are converted to raw API URLs.
- **Bitbucket** `/src/` paths are converted to raw download URLs.
- **Google Drive** links are handled via `download_google_drive_file()` (extracts file ID from URL).
- Fallback: if URL conversion fails, the original URL is returned unchanged.

### 7. Auth Fallback Order (Kaggle)

`get_auth()` checks sources in this order:
1. `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars
2. `~/.kaggle/kaggle.json`
3. `.env` file via `python-dotenv` (project root)
4. `kagglehub` SDK credentials (`get_kagglehub_credentials()`)

**Known issue** (per `fail.md`): the `.env` fallback may fail if `python-dotenv` is not installed or `.env` is not in the project root. The kagglehub SDK fallback can interfere with dotenv loading — consider reordering or documenting the precedence explicitly.

### 8. Notebook Conversion Pipeline

When deploying, notebooks go through:
1. `convert_notebook_to_script()` — uses `nbconvert` if available, else manual cell extraction
2. `clean_jupyter_magic_commands()` — strips Jupyter magics (`%%`, `%`, `get_ipython()`)
3. `_detect_model_requirements()` — scans for imports → generates `requirements.txt`
4. Packaging: `package_for_gcf()` / `package_for_aws()` / `package_for_modal()`
5. Deployment: `deploy_to_gcloud()` / `deploy_to_aws_lambda()` / `deploy_to_modal()`

### 9. `kaggle_cli_cmd()` Internal Helper

Many kernel operations delegate to the official `kaggle` CLI under the hood:
```python
kaggle_cli_cmd(["kernels", "list", ...])  # kernels_list()
kaggle_cli_cmd(["kernels", "output", ref, ...])  # _download_outputs()
```
If the `kaggle` CLI is not installed, these commands fail silently or raise `FileNotFoundError`.

### 10. `--validate` Flag

In addition to `python3 runna.py doctor`, the CLI also accepts:
```bash
python3 runna.py --validate  # runs validate_setup() directly
```
Note: `doctor` (via `validate_setup()`) checks kaggle CLI, gcloud CLI, and auth, but does **not** check `aws`, `modal`, `nbconvert`, or `.env` loading. The standalone `doctor.py` script checks Python version, kaggle package, credentials, and API connection — but also does not check cloud providers beyond gcloud.

## Discovered Code Patterns & Known Issues

### Exception Handling

`runna.py` has **71 `except Exception` blocks** and **18 silent `except Exception:` blocks** (no `as e`, no logging). Key locations:
- Lines 701, 717, 729, 741: URL conversion helpers — silently catch all exceptions, returning original URL on failure.
- Line 863: `download_remote_notebook` fallback — silently catches, making remote download failures invisible.
- Line 970: `_load_endpoints()` — silently catches JSON parse / file read errors.
- Lines 1173, 1278: batch / deploy paths — silently catch errors.
- Line 2097: `deploy_to_modal` — silently catches deployment failures.

This is a live bug source: users see no error when these paths fail (see `fail.md` for examples).

### Monolith Structure

`runna.py` is **3,112 lines**, **99 functions**, **3 dataclasses**. The `main()` argparser is **~380 lines**. This makes targeted changes risky — partial refactors have been attempted and reverted (PRs #4–#8).

### Test Coverage

Only shell-script tests exist (`test_app_library.sh`, `test_modal.sh`, `test_send.sh`). No pytest, no unit assertions. A prior pytest suite was lost when the `notebook_ml_orchestrator` refactor was reverted (PR #8).

## Test Scripts

```bash
./test_app_library.sh   # app library CRUD
./test_modal.sh          # Modal deployment (GPU, timeout)
./test_send.sh           # send/chat functionality
```

## Key Dependencies

From `requirements.txt`: flask, kaggle, python-dotenv, kagglehub, nbformat, nbconvert, functions-framework, modal, requests.

## Notebook Convention for Deployment

Notebooks should define a `process_request(data)` function. The generated serverless handler calls this function; if absent, falls back to `deploy_model.predict()` or echo.

## .gitignore

Covers: `.env*`, `env`, `venv`, `.kaggle`. Endpoint state (`.kaggle_state/`) and `__pycache__/` are not ignored — TODO: consider adding.

## Automation Agent Notes

- **Pre-flight check**: Before opening a PR, search `gh pr list` for existing PRs targeting the same files. The repo has a history of ~280 duplicate PRs (PRs #280–#329) for the same exception-handling fix.
- **Stale PRs**: PRs #197 and #66 are still open but duplicate already-merged PRs #12, #302, #307.
- **Branch naming**: Use `tembo/` prefix for agent-created branches.
- **Monolith caution**: Changes to `runna.py` should be surgical and tested — the file is 3,112 lines and tightly coupled.

## TODO

- Add `.kaggle_state/` and `__pycache__/` to `.gitignore`
- Verify `jupyterapi_nbrunner/` module purpose and document if actively used (directory is currently empty)
- Enhance `doctor` / `validate_setup()` to check `aws`, `modal`, `nbconvert`, and `.env` loading
- Replace silent `except Exception:` blocks with specific exception types + logging (see Discovered Code Patterns)
- Add pytest test suite to replace shell-script-only coverage
