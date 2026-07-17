# AGENTS.md — gPU Project Reference

Auto-maintained reference for workflows, commands, and conventions in this repo.

---

## Project Overview

Kaggle Notebook Automation & Deployment Tool. The main CLI is **`runna.py`** (~3100 lines). It wraps the Kaggle CLI/SDK and adds deployment to GCP Cloud Functions, AWS Lambda, and Modal.com, plus an app library, endpoint registry, and text send/chat.

### Key Files

| File | Purpose |
|---|---|
| `runna.py` | Main CLI entry point — all subcommands live here |
| `app_library.py` | App library CRUD (JSON index at `apps/library.json`) |
| `deploy_model.py` | Model loading, Kaggle/HF download helpers |
| `doctor.py` | Standalone environment validator (Python, kaggle pkg, creds, API) |
| `modal_deploy.py` | Standalone Modal.com deployment template |
| `deploy.sh` | Shell script for GCP Cloud Functions deploy |
| `requirements.txt` | Python dependencies |
| `kernel-metadata.json` | Kaggle kernel metadata for push |
| `apps/*.py` | Built-in app templates (6 apps) |
| `apps/library.json` | App library index |
| `examples/` | Sample notebooks and Modal example |

---

## Quick Setup

```bash
pip install -r requirements.txt
python3 runna.py doctor   # verify environment (alias: python3 runna.py --validate)
```

---

## CLI Command Reference (`python3 runna.py`)

### Notebook Operations

| Command | Usage | Notes |
|---|---|---|
| `list` | `runna.py list [--user USER] [--search TERM]` | Lists kernels via Kaggle API |
| `pull` | `runna.py pull USER/NOTEBOOK [--dest DIR]` | Downloads .ipynb from Kaggle; also accepts Kaggle URLs |
| `push` | `runna.py push PATH` | Uploads notebook to Kaggle |
| `create` | `runna.py create [INPUT]` | Creates kernel from local file, remote URL, or blank if no input |
| `run` | `runna.py run [INPUT] [--deploy]` | Pull+push+run; interactive selection if no input; `--deploy` triggers GCP deploy after run |
| *(shortcut)* | `runna.py --validate` | Top-level flag, alias for `doctor` |

### Deployment

| Command | Usage | Notes |
|---|---|---|
| `deploy` | `runna.py deploy PATH --gcp-project ID [opts]` | Deploy to GCP Cloud Functions |
| `deploy-aws` | `runna.py deploy-aws PATH --role-arn ARN [opts]` | Deploy to AWS Lambda; also supports `--function-name`, `--region`, `--memory`, `--timeout`, `--save-name` |
| `deploy-modal` | `runna.py deploy-modal PATH [--gpu TYPE] [--secrets ...] [--timeout N]` | Deploy to Modal.com |
| `package-aws` | `runna.py package-aws PATH` | Package as Lambda zip (no deploy) |
| `serve-local` | `runna.py serve-local PATH [--run] [--port 8080]` | Package for local testing; `--run` starts functions-framework |

### GCP Deploy Options (shared by `deploy` and `run --deploy`)

```
--gcp-project ID       # Required
--region REGION        # Default: us-central1 (or env REGION)
--function-name NAME   # Default: auto-generated timestamp
--memory MEMORY        # Default: 512MB
--timeout TIMEOUT      # Default: 540s
--save-name NAME       # Register endpoint for later use
--private              # Require auth (default is --public)
```

### Modal Deploy Options

```
--gpu TYPE             # T4 | A10G | A100 | None (default: None)
--secrets NAME ...     # Modal secret names to attach
--timeout SECONDS      # Default: 300
--save-name NAME       # Register endpoint for later use
```

### App Library

| Command | Usage | Notes |
|---|---|---|
| `app-list` | `runna.py app-list` | Shows name, GPU, deployed status, description |
| `app-show` | `runna.py app-show NAME` | Display app source and metadata |
| `app-deploy` | `runna.py app-deploy NAME [--gpu TYPE] [--secrets ...] [--timeout N]` | Deploy app via Modal |
| `app-add` | `runna.py app-add NAME --file FILE [--desc ...] [--tags ...] [--gpu TYPE]` | Add to library |
| `app-update` | `runna.py app-update NAME --file FILE` | Update app code |
| `app-delete` | `runna.py app-delete NAME` | Remove from library |

Built-in apps: `image-classifier` (T4), `text-generator` (T4), `llm-chat` (A10G), `web-scraper` (CPU), `batch-processor` (CPU), `scheduled-task` (CPU)

### Endpoint & Interaction

| Command | Usage | Notes |
|---|---|---|
| `endpoints` | `runna.py endpoints` | List saved endpoints from `.kaggle_state/endpoints.json` |
| `call` | `runna.py call NAME_OR_URL --json '{}' [--method POST] [--json-file FILE]` | Send JSON to endpoint (inline or from file); **prompts interactively for JSON if neither `--json` nor `--json-file` is given** |
| `send` | `runna.py send ENDPOINT TEXT [--field FIELD]` | Send text; default field is `"text"` |
| `chat` | `runna.py chat ENDPOINT [--field FIELD]` | Interactive chat loop (Ctrl+C to exit) |

### Utility

| Command | Usage | Notes |
|---|---|---|
| `batch` | `runna.py batch FILE --operation {download,deploy,run} [--output-dir DIR] [--parallel]` | Process list of notebooks; `--parallel` for concurrent execution |
| `preprocess` | `runna.py preprocess PATH [--clean-metadata] [--remove-outputs] [--scan-security] [--optimize-imports] [--output PATH]` | Clean notebooks; `--output` writes to a different file |
| `doctor` | `runna.py doctor` | Validate Python, kaggle pkg, credentials, API connection |

---

## Workflows

### 1. Pull → Run → Deploy (Kaggle to Serverless API)

```bash
# Pull a notebook
python3 runna.py pull username/kernel-name --dest ./notebooks

# Run on Kaggle, then deploy to GCP
python3 runna.py run username/kernel-name --deploy \
  --gcp-project my-project --save-name my-api

# Or deploy an already-pulled notebook
python3 runna.py deploy ./notebooks/kernel-name \
  --gcp-project my-project --save-name my-api
```

### 2. App Library: Browse → Deploy → Test

```bash
python3 runna.py app-list
python3 runna.py app-show llm-chat
python3 runna.py app-deploy llm-chat
python3 runna.py send llm-chat "Hello"
python3 runna.py chat llm-chat
```

### 3. Local Development

```bash
python3 runna.py serve-local ./notebook-dir --run --port 8080
curl -X POST http://localhost:8080 -H "Content-Type: application/json" -d '{"features": [1,2,3]}'
```

### 4. Multi-Platform Deploy

```bash
# GCP
python3 runna.py deploy ./nb --gcp-project ID --save-name gcp-api
# AWS
python3 runna.py deploy-aws ./nb --role-arn ARN --save-name aws-api
# Modal (with GPU)
python3 runna.py deploy-modal ./nb --gpu A10G --secrets api-key --save-name modal-api
```

### 5. Batch Processing

```bash
# Create notebooks.txt with one kernel ref per line
python3 runna.py batch notebooks.txt --operation download --output-dir ./downloads
python3 runna.py batch notebooks.txt --operation deploy --output-dir ./deployments
```

### 6. Custom App → Library → Deploy

```bash
python3 runna.py app-add my-app --file my_app.py --description "My app" --gpu T4
python3 runna.py app-deploy my-app --gpu A100
python3 runna.py endpoints
python3 runna.py call my-app --json '{"input": "test"}'
```

---

## Authentication Chain

`runna.py` resolves Kaggle credentials in this order (see `get_auth()`):

1. **Environment variables**: `KAGGLE_USERNAME` + `KAGGLE_KEY`
2. **`~/.kaggle/kaggle.json`**: Standard Kaggle CLI config
3. **`.env` file** in project root (requires `python-dotenv`; loaded with `override=True`)
4. **kagglehub SDK** (last resort; calls `kagglehub.login()`)

Set up via:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
# OR
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# Verify:
python3 runna.py doctor
```

Cloud providers: `gcloud auth login`, `aws configure`, `modal token new`.

---

## Input Type Resolution (`determine_input_type`)

The `create` and `run` commands auto-detect input type:

| Input | Detected As | Behavior |
|---|---|---|
| `username/kernel-name` | `kaggle_ref` | Pull from Kaggle |
| `https://kaggle.com/code/...` | `kaggle_url` | Parse URL, pull from Kaggle |
| Any URL ending in `.ipynb` | `remote_notebook` | Download (host-agnostic catch-all) |
| `https://github.com/...` | `remote_notebook` | Download from GitHub (auto-converts to raw URL) |
| `https://gist.github.com/...` | `remote_notebook` | Download from GitHub Gist |
| `https://gitlab.com/...` | `remote_notebook` | Download from GitLab (raw conversion) |
| `https://bitbucket.org/...` | `remote_notebook` | Download from Bitbucket (raw conversion) |
| `https://drive.google.com/...` | `remote_notebook` | Download from Google Drive |
| `https://docs.google.com/...` | `remote_notebook` | Download from Google Docs |
| `./local/path` or `file.ipynb` | `local_file` / `local_dir` | Use directly |
| *(no input to `run`)* | — | Interactive selection from popular kernels |

---

## Notebook Convention for Deployment

Notebooks intended for deployment should define a `process_request(data)` function:

```python
def process_request(data):
    """Called by the serverless handler."""
    features = data.get('features', [])
    result = sum(features)
    return {'result': result}
```

The generated handler (`deploy/main.py` or `modal-deploy/modal_app.py`) tries:
1. `process_request(data)` from notebook code
2. `deploy_model.predict(features)` if `deploy_model.py` exists
3. Echo the input as fallback

---

## Dependency Auto-Detection

`_detect_model_requirements()` in `runna.py` scans `deploy_model.py` (checked at both the project root and CWD) for imports and adds:
- `torch` → PyTorch
- `sklearn` → scikit-learn
- `numpy`, `pandas`, `requests`
- `kagglehub`, `huggingface_hub`, `python-dotenv`

These are merged into `deploy/requirements.txt` alongside base deps (`flask>=2.0.0`, `requests>=2.25.0`, `functions-framework>=3.5.0`) by `create_requirements_file()`.

## Jupyter Magic Auto-Cleaning

During deployment, `clean_jupyter_magic_commands()` automatically strips IPython-specific syntax that would cause `SyntaxError` in production:
- Cell magic (`%%writefile`, `%%time`, `%%bash`)
- Line magic (`%matplotlib`, `%load_ext`, `%pip`)
- `get_ipython()` calls
- IPython cell markers (`# In[1]:`)

This is applied in `convert_notebook_to_script()` and fallback extraction paths.

---

## Known Issues (from `fail.md`)

- **Kernel push 409 Conflict**: Title-to-ID slug mismatch causes `kaggle kernels push` to fail with "Your kernel title does not resolve to the specified id". The auto-generated metadata may produce incompatible slugs.
- **`.env` loading in venv**: `python-dotenv` may not find the `.env` file depending on venv configuration; environment variables or `~/.kaggle/kaggle.json` are more reliable.
- **`create` command namespace error**: Some `create` code paths hit `'Namespace' object has no attribute 'path'`.

---

## Generated File Locations

| Artifact | Path | Created By |
|---|---|---|
| GCF deploy dir | `<notebook_dir>/deploy/` | `deploy`, `run --deploy`, `serve-local` |
| Lambda package | `<notebook_dir>/aws-lambda.zip` | `package-aws`, `deploy-aws` |
| Modal app | `<notebook_dir>/modal-deploy/modal_app.py` | `deploy-modal` |
| Endpoint registry | `.kaggle_state/endpoints.json` | Any `--save-name` deploy |
| App library | `apps/library.json` + `apps/<name>.py` | `app-add`, `app-update`, `app-delete` |

---

## GPU Pricing (Modal.com)

| Type | $/hr | Best For |
|---|---|---|
| T4 | 0.60 | Inference |
| A10G | 1.10 | Balanced |
| A100 | 4.00 | Training |

## Free Tier Reference

| Platform | Free Allowance |
|---|---|
| GCP Cloud Functions | 2M requests/month |
| AWS Lambda | 1M requests/month |
| Modal.com | $30 free credit/month |

---

## Test Scripts

| Script | Purpose |
|---|---|
| `test_app_library.sh` | App library CRUD round-trip |
| `test_modal.sh` | Modal deploy with GPU/timeout options |
| `test_send.sh` | Send/chat dry-run (requires real deploy for live test) |

---

## Key Dependencies

From `requirements.txt`: flask, kaggle, python-dotenv, kagglehub, nbformat, nbconvert, functions-framework, modal, requests.

---

## .gitignore

Covers: `.env*`, `env`, `venv`, `.kaggle`. Endpoint state (`.kaggle_state/`) and `__pycache__/` are not ignored — TODO: consider adding.

---

## TODO

- Automated `run → deploy` pipeline (pull, push, wait for complete, then deploy — currently manual)
- Fix kernel-metadata.json slug generation to avoid 409 push conflicts
- Add `.kaggle_state/` and `__pycache__/` to `.gitignore`
- Verify `jupyterapi_nbrunner/` module purpose and document if actively used
