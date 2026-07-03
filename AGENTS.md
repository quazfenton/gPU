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

Three sources checked in order:
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

## TODO

- Add `.kaggle_state/` and `__pycache__/` to `.gitignore`
- Verify `jupyterapi_nbrunner/` module purpose and document if actively used
