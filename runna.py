#!/usr/bin/env python3
"""
Enhanced Kaggle CLI tool with robust error handling and deployment capabilities.
Supports downloading, running, and deploying notebooks from Kaggle to Google Cloud Functions.
"""

import os
import sys
import json
import shutil
import argparse
import logging
import tempfile
import requests
import re
import zipfile
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.kaggle.com/api/v1"

# ----------------------------
# Authentication & Setup
# ----------------------------


def get_auth():
    """Get Kaggle API credentials with multiple fallback methods."""
    # 1. Try environment variables first (most reliable)
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        logger.info("Using credentials from environment variables")
        return (username, key)

    # 2. Try official kaggle.json file
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.exists():
        try:
            with open(kaggle_json_path, "r") as f:
                creds = json.load(f)
                logger.info("Using credentials from ~/.kaggle/kaggle.json")
                return (creds["username"], creds["key"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid kaggle.json: {e}")

    # 3. Try .env file in current directory
    env_path = Path(".env")
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            # Ensure .env takes effect even if variables are already set in this process
            load_dotenv(dotenv_path=str(env_path), override=True)
            username = os.environ.get("KAGGLE_USERNAME")
            key = os.environ.get("KAGGLE_KEY")
            if username and key:
                logger.info("Using credentials from .env file")
                return (username, key)
        except ImportError:
            logger.warning("python-dotenv not installed, skipping .env file")

    # 4. Try kagglehub SDK as last resort
    try:
        import kagglehub

        # kagglehub uses same credential sources but might have different loading
        logger.info("Attempting to use kagglehub SDK credentials")
        # This will raise an exception if no credentials found
        kagglehub.login()
        # If we get here, credentials exist - try to extract them
        return get_kagglehub_credentials()
    except ImportError:
        logger.warning("kagglehub not installed")
    except Exception as e:
        logger.warning(f"kagglehub authentication failed: {e}")

    raise RuntimeError(
        "Kaggle API credentials not found. Please:\n"
        "1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
        "2. Place kaggle.json in ~/.kaggle/\n"
        "3. Create .env file with KAGGLE_USERNAME and KAGGLE_KEY\n"
        "4. Install and configure kagglehub: pip install kagglehub"
    )


def get_kagglehub_credentials():
    """Extract credentials from kagglehub if available."""
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.exists():
        with open(kaggle_json_path, "r") as f:
            creds = json.load(f)
            return (creds["username"], creds["key"])
    raise RuntimeError("Could not extract kagglehub credentials")


def setup_kaggle_auth():
    """Ensure Kaggle CLI authentication is set up."""
    try:
        username, key = get_auth()
        # Set environment variables for kaggle CLI
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        return True
    except Exception as e:
        logger.error(f"Authentication setup failed: {e}")
        return False


# ----------------------------
# Official Kaggle CLI Wrappers
# ----------------------------


def kaggle_cli_cmd(cmd_args, capture_output=True):
    """Execute kaggle CLI command with proper error handling."""
    if not setup_kaggle_auth():
        raise RuntimeError("Authentication not configured")

    full_cmd = ["kaggle"] + cmd_args
    logger.info(f"Executing: {' '.join(full_cmd)}")

    try:
        return subprocess.run(
            full_cmd, capture_output=capture_output, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise RuntimeError(f"Kaggle CLI command failed: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "Kaggle CLI not found. Install with: pip install kaggle"
        ) from e


# ----------------------------
# HTTP API Wrappers (Fallback)
# ----------------------------


def _prepare_request_kwargs(auth, data=None, files=None, json_data=None):
    """Prepare kwargs for HTTP requests."""
    kwargs = {"auth": auth}
    if json_data:
        kwargs["json"] = json_data
    elif files:
        kwargs["files"] = files
        kwargs["data"] = data or {}
    else:
        kwargs["data"] = data
    return kwargs


def kaggle_get(path, params=None, stream=False):
    """Generic GET request wrapper."""
    url = f"{BASE_URL}{path}"
    try:
        auth = get_auth()
        r = requests.get(url, auth=auth, params=params, stream=stream)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        logger.error(f"GET {url} failed: {e}")
        raise RuntimeError(f"API request failed: {e}") from e


def kaggle_post(path, data=None, files=None, json_data=None):
    """Generic POST request wrapper."""
    url = f"{BASE_URL}{path}"
    try:
        auth = get_auth()
        kwargs = _prepare_request_kwargs(auth, data, files, json_data)
        r = requests.post(url, **kwargs)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        logger.error(f"POST {url} failed: {e}")
        raise RuntimeError(f"API request failed: {e}") from e


# ----------------------------
# Kernels API Functions
# ----------------------------


def _build_kernels_list_cmd(username=None, search=None):
    """Build kernels list command."""
    cmd = ["kernels", "list"]
    if username:
        cmd.extend(["--user", username])
    if search:
        cmd.extend(["--search", search])
    return cmd


def _parse_kernels_output(output):
    """Parse CLI kernels list output into structured format."""
    lines = [ln for ln in output.strip().split("\n") if ln.strip()]
    kernels = []
    if not lines:
        return kernels
    # Skip header and any dashed separator lines
    start_idx = 0
    # Skip the first line (header row)
    if start_idx < len(lines):
        start_idx += 1
    # Skip any subsequent lines that are just dashes
    while start_idx < len(lines) and set(lines[start_idx].strip()) == {"-"}:
        start_idx += 1
    for line in lines[start_idx:]:
        if not line.strip() or set(line.strip()) == {"-"}:
            continue
        parts = line.split()
        if len(parts) >= 3:
            kernels.append(
                {
                    "ref": parts[0],
                    "title": " ".join(parts[1:-1]),
                    "lastRunTime": parts[-1] if parts[-1] != "never" else None,
                }
            )
    return kernels


def _get_kaggle_api():
    """Try to initialize KaggleApi if available (optional). Returns authenticated api or None.
    Note: this may fail if this file name shadows the official 'kaggle' package.
    """
    try:
        import importlib

        mod = importlib.import_module("kaggle.api.kaggle_api_extended")
        KaggleApi = getattr(mod, "KaggleApi")
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        logger.debug(f"KaggleApi unavailable or failed to init: {e}")
        return None


def kernels_list(username=None, search=None):
    """List kernels for a user. Prefer KaggleApi when available, fallback to CLI then HTTP."""
    # 1) Prefer Python API when importable (user may rename this file later to avoid shadowing)
    api = _get_kaggle_api()
    if api:
        try:
            user_to_list = username or get_auth()[0]
            kernels = api.kernels_list(user=user_to_list, search=search)
            out = []
            for k in kernels:
                ref = getattr(k, "ref", None)
                if not ref:
                    author = getattr(k, "author", user_to_list)
                    slug = getattr(k, "slug", "")
                    ref = f"{author}/{slug}" if slug else author
                out.append(
                    {
                        "ref": ref,
                        "title": getattr(k, "title", ""),
                        "lastRunTime": str(getattr(k, "last_run_time", "")),
                    }
                )
            return out
        except Exception as e:
            logger.warning(f"KaggleApi list failed, falling back to CLI: {e}")

    # 2) CLI
    try:
        cmd = _build_kernels_list_cmd(username, search)
        result = kaggle_cli_cmd(cmd)
        return _parse_kernels_output(result.stdout)
    except Exception as e:
        logger.warning(f"CLI list failed, trying HTTP API: {e}")
        # 3) HTTP
        user_to_list = username or get_auth()[0]
        params = {"user": user_to_list}
        if search:
            params["search"] = search
        r = kaggle_get("/kernels/list", params=params)
        return r.json()


def _find_pulled_directory(dest_path, ref_clean):
    """Find the directory where kernel was pulled."""
    pulled_dir = dest_path / ref_clean
    if pulled_dir.exists():
        return pulled_dir

    # Look for any new directories containing the ref
    for item in dest_path.iterdir():
        if item.is_dir() and ref_clean in item.name:
            return item
    return None


def _extract_kernel_content(content, out_dir, kernel_slug):
    """Extract kernel content from response."""
    if content.startswith(b"PK"):  # ZIP file
        zip_path = out_dir / "temp.zip"
        with open(zip_path, "wb") as f:
            f.write(content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(out_dir)
        zip_path.unlink()
    else:
        # Plain text/JSON content
        notebook_path = out_dir / f"{kernel_slug}.ipynb"
        with open(notebook_path, "wb") as f:
            f.write(content)


def kernel_pull(ref, dest="."):
    """Pull kernel using official CLI, including metadata for re-push.
    Ensures artifacts are placed under a dedicated subdirectory and validates the pulled notebook file.
    """
    try:
        dest_path = Path(dest)
        ref_clean = ref.replace("/", "_")
        out_dir = dest_path / ref_clean
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use official CLI with metadata into dedicated folder
        kaggle_cli_cmd(["kernels", "pull", ref, "--path", str(out_dir), "--metadata"])

        # Prefer metadata to locate the code file
        meta = out_dir / "kernel-metadata.json"
        if meta.exists():
            try:
                meta_json = json.loads(meta.read_text(encoding="utf-8"))
                code_file = meta_json.get("code_file")
                if code_file:
                    nb_path = out_dir / code_file
                    if nb_path.exists() and nb_path.stat().st_size > 0:
                        logger.info(f"Successfully pulled kernel to → {out_dir}")
                        return out_dir
                    else:
                        logger.warning(
                            "Pulled notebook is missing or empty, retrying via HTTP API..."
                        )
                        return kernel_pull_alternative(ref, dest)
            except Exception as me:
                logger.warning(
                    f"Could not read kernel-metadata.json ({me}); attempting fallback detection"
                )

        # If no metadata, try common patterns
        username, kernel_slug = ref.split("/", 1)
        expected_ipynb = out_dir / f"{kernel_slug}.ipynb"
        if expected_ipynb.exists() and expected_ipynb.stat().st_size > 0:
            logger.info(f"Successfully pulled kernel to → {out_dir}")
            return out_dir

        # Final fallback: HTTP API pull
        logger.warning("Could not validate pull artifacts; retrying via HTTP API...")
        return kernel_pull_alternative(ref, dest)

    except Exception as e:
        logger.error(f"Failed to pull kernel: {e}")
        # Try alternative method
        return kernel_pull_alternative(ref, dest)


def kernel_pull_alternative(ref, dest="."):
    """Alternative pull method using direct API."""
    username, kernel_slug = ref.split("/", 1)
    params = {"userName": username, "kernelSlug": kernel_slug}

    try:
        r = kaggle_get("/kernels/pull", params=params, stream=True)

        out_dir = Path(dest) / ref.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract content
        _extract_kernel_content(r.content, out_dir, kernel_slug)

        logger.info(f"Successfully pulled kernel to → {out_dir}")
        return out_dir

    except Exception as e:
        logger.error(f"Alternative pull method failed: {e}")
        raise RuntimeError(f"Failed to pull kernel {ref}") from e


def _setup_work_directory(source_path):
    """Setup working directory for kernel push."""
    if source_path.is_dir():
        work_dir = source_path
        is_temp = False
    else:
        # Single file - create temp directory
        work_dir = Path(tempfile.mkdtemp())
        shutil.copy(source_path, work_dir)
        is_temp = True

    meta_file = work_dir / "kernel-metadata.json"
    return work_dir, meta_file, is_temp


def _handle_push_error(e, work_dir):
    """Handle kernel push errors."""
    if "already exists" in str(e.stderr) or "409" in str(e.stderr):
        logger.warning("Kernel already exists, trying version update...")
        return kernel_push_version(work_dir)
    elif "title does not resolve" in str(e.stderr):
        logger.warning("Title resolution issue, regenerating metadata...")
        # Try to fix metadata and retry
        return _retry_push_with_new_metadata(work_dir)
    else:
        logger.error(f"Push failed: {e.stderr}")
        raise RuntimeError(f"Kernel push failed: {e}") from e


def _retry_push_with_new_metadata(work_dir):
    """Retry push with regenerated metadata."""
    try:
        # Remove old metadata and create new one
        meta_file = work_dir / "kernel-metadata.json"
        if meta_file.exists():
            meta_file.unlink()

        # Find the main code file
        code_files = list(work_dir.glob("*.ipynb")) + list(work_dir.glob("*.py"))
        if code_files:
            create_kernel_metadata(work_dir, code_files[0])
            cmd = ["kernels", "push", "--path", str(work_dir)]
            result = kaggle_cli_cmd(cmd)
            logger.info("Kernel push successful after metadata regeneration!")
            logger.info(result.stdout)
            return True
    except Exception as e:
        logger.error(f"Retry with new metadata failed: {e}")
        raise


def kernel_push(path, message=None):
    """Push kernel using official CLI with proper metadata. Returns kernel ref id."""
    source_path = Path(path)

    if not source_path.exists():
        raise FileNotFoundError(f"Path not found: {source_path}")

    work_dir, meta_file, is_temp = _setup_work_directory(source_path)

    try:
        if not meta_file.exists():
            create_kernel_metadata(work_dir, source_path)

        # Read ref from metadata
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        ref = metadata.get("id")

        # Use official CLI
        cmd = ["kernels", "push", "--path", str(work_dir)]
        result = kaggle_cli_cmd(cmd)

        logger.info("Kernel push successful!")
        logger.info(result.stdout)
        return ref

    except subprocess.CalledProcessError as e:
        _handle_push_error(e, work_dir)
        # After handling (e.g., version push), return ref again
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        return metadata.get("id")
    finally:
        # Cleanup temp directory if created
        if is_temp and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def kernel_push_version(work_dir):
    """Push new version of existing kernel."""
    try:
        cmd = ["kernels", "push", "--path", str(work_dir)]
        result = kaggle_cli_cmd(cmd)

        logger.info("Kernel version update successful!")
        logger.info(result.stdout)
        return True
    except Exception as e:
        logger.error(f"Version push failed: {e}")
        raise


def _find_code_file(work_dir, source_path):
    """Find the main code file for the kernel."""
    if source_path.is_file():
        return source_path.name

    code_files = list(work_dir.glob("*.ipynb")) + list(work_dir.glob("*.py"))
    if not code_files:
        raise ValueError("No .ipynb or .py files found")
    return code_files[0].name


def _generate_title_from_filename(code_file):
    """Generate a readable title from filename."""
    return (
        code_file.replace(".ipynb", "")
        .replace(".py", "")
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


def create_kernel_metadata(work_dir, source_path):
    """Create proper kernel metadata JSON."""
    work_dir = Path(work_dir)
    source_path = Path(source_path)

    # Find the main code file
    code_file = _find_code_file(work_dir, source_path)

    # Generate metadata
    username = get_auth()[0]
    slug = generate_unique_slug(code_file)

    # Make the title resolve to the slug to avoid CLI warnings/errors
    title = slug.replace("-", " ").title()

    # Create metadata in official format
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook" if code_file.endswith(".ipynb") else "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }

    meta_file = work_dir / "kernel-metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created metadata for kernel: {metadata['id']}")


def generate_unique_slug(filename):
    """Generate unique slug for kernel."""
    base = filename.replace(".ipynb", "").replace(".py", "")
    base = re.sub(r"[^a-zA-Z0-9\-]", "-", base).lower()
    base = re.sub(r"-+", "-", base).strip("-")

    # Add timestamp for uniqueness
    timestamp = str(int(time.time()))
    return f"{base}-{timestamp}"


def kernel_status(ref):
    """Get kernel status."""
    try:
        cmd = ["kernels", "status", ref]
        result = kaggle_cli_cmd(cmd)

        # Parse status from output
        status_text = result.stdout.strip()
        return {"status": status_text}
    except Exception as e:
        logger.warning(f"Status check failed: {e}")
        return {"status": "unknown"}


# ----------------------------
# Utility Functions
# ----------------------------


def validate_url(url):
    """Validate and sanitize URL."""
    if not url or not isinstance(url, str):
        return False, "URL is empty or not a string"

    # Basic URL format validation
    if not url.startswith(("http://", "https://")):
        return False, "URL must start with http:// or https://"

    # Check for potentially dangerous characters/sequences
    dangerous_patterns = ["..", "://..", ";", "&", "|", "$(", "`"]
    for pattern in dangerous_patterns:
        if pattern in url:
            return False, f"URL contains potentially dangerous pattern: {pattern}"

    # Validate URL length
    if len(url) > 2000:  # Standard URL length limit
        return False, "URL is too long (max 2000 characters)"

    return True, "Valid URL"


def determine_input_type(input_str):
    """Determine type of input string."""
    if not input_str:
        return "none"

    if input_str.startswith(("http://", "https://")):
        is_valid, msg = validate_url(input_str)
        if not is_valid:
            logger.warning(f"Invalid URL: {msg}")
            return "invalid_url"

        if "kaggle.com" in input_str:
            return "kaggle_url"
        elif input_str.endswith(".ipynb"):
            return "remote_notebook"
        elif any(
            host in input_str
            for host in [
                "github.com",
                "gist.github.com",
                "gitlab.com",
                "bitbucket.org",
                "drive.google.com",
                "docs.google.com",
            ]
        ):
            return "remote_notebook"
        return "other_url"

    # Check for Kaggle reference format (username/kernel-name)
    if (
        "/" in input_str
        and not input_str.startswith((".", "/"))
        and len(input_str.split("/")) == 2
    ):
        # Additional validation for Kaggle reference format
        parts = input_str.split("/")
        if len(parts) == 2 and len(parts[0]) > 0 and len(parts[1]) > 0:
            # Check for valid characters (alphanumeric, hyphens, underscores)
            import re

            if re.match(r"^[\w-]+$", parts[0]) and re.match(r"^[\w-]+$", parts[1]):
                return "kaggle_ref"
        return "invalid_kaggle_ref"


def run_notebook_in_kaggle(ref, timeout_sec=1800, poll_every=15):
    """Wait for a Kaggle kernel to complete."""
    try:
        start = time.time()
        while True:
            status = kernel_status(ref)
            logger.info(f"Kernel status: {status}")
            status_text = str(status.get("status", "")).lower()
            if "complete" in status_text:
                return True
            if "error" in status_text or "failed" in status_text:
                raise RuntimeError(f"Kernel run failed: {status}")
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Timed out waiting for kernel to complete: {ref}")
            time.sleep(poll_every)
    except Exception as e:
        logger.error(f"Failed while waiting for Kaggle run: {e}")
        raise


def parse_kaggle_url(url):
    """Extract kernel reference from Kaggle URL."""
    patterns = [r"kaggle\.com/code/([\w-]+/[\w-]+)", r"kaggle\.com/([\w-]+/[\w-]+)"]

    for pattern in patterns:
        if match := re.search(pattern, url):
            return match[1]
    return None


def _maybe_github_blob_to_raw(url: str) -> str:
    """Convert a GitHub blob URL to raw if applicable."""
    try:
        if "github.com" in url and "/blob/" in url:
            parts = url.split("github.com/")[-1]
            owner_repo, rest = parts.split("/", 1)
            owner, repo = (
                owner_repo.split("/") if "/" in owner_repo else owner_repo.split("/")
            )
            # Rebuild raw URL
            raw = url.replace(
                "https://github.com/", "https://raw.githubusercontent.com/"
            ).replace("/blob/", "/")
            return raw
    except Exception:
        pass
    return url


def _convert_github_gist_to_raw(url: str) -> str:
    """Convert a GitHub Gist URL to raw if applicable."""
    try:
        if "gist.github.com" in url:
            # Convert gist URL to raw format
            # Example: https://gist.github.com/username/abc123 -> https://gist.githubusercontent.com/username/abc123/raw
            parts = url.replace("https://", "").replace("http://", "").split("/")
            if len(parts) >= 3:
                username = parts[1]
                gist_id = parts[2].split("#")[0]  # Remove anchor
                return f"https://gist.githubusercontent.com/{username}/{gist_id}/raw"
    except Exception:
        pass
    return url


def _convert_gitlab_to_raw(url: str) -> str:
    """Convert a GitLab URL to raw if applicable."""
    try:
        if "gitlab.com" in url and "/-/blob/" in url:
            # Convert GitLab URL to raw format
            raw = url.replace("/-/blob/", "/-/raw/")
            return raw
    except Exception:
        pass
    return url


def _convert_bitbucket_to_raw(url: str) -> str:
    """Convert a Bitbucket URL to raw if applicable."""
    try:
        if "bitbucket.org" in url and "/src/" in url:
            # Convert Bitbucket URL to raw format
            raw = url.replace("/src/", "/raw/")
            return raw
    except Exception:
        pass
    return url


def _normalize_notebook_url(url: str) -> str:
    """Normalize various notebook hosting services URLs to their raw content URLs."""
    # Apply transformations in order
    url = _maybe_github_blob_to_raw(url)
    url = _convert_github_gist_to_raw(url)
    url = _convert_gitlab_to_raw(url)
    url = _convert_bitbucket_to_raw(url)
    return url


def download_google_drive_file(url):
    """Download notebook from Google Drive if possible."""
    try:
        # Extract file ID from Google Drive URL
        file_id = None
        if "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
        elif "/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]

        if not file_id:
            logger.error(f"Could not extract file ID from Google Drive URL: {url}")
            return None

        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".ipynb", delete=False, encoding="utf-8"
        )
        temp_file.write(response.text)
        temp_file.close()

        logger.info(f"Downloaded notebook from Google Drive {url} to {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Failed to download Google Drive file {url}: {e}")
        return None


def download_remote_notebook(url):
    """Download notebook from various URL sources with enhanced error handling."""
    try:
        # Normalize URL to get raw content
        normalized_url = _normalize_notebook_url(url)

        # Try multiple methods with different headers
        headers_list = [
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            },
            {},
        ]

        response = None
        for headers in headers_list:
            try:
                response = requests.get(normalized_url, timeout=60, headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                continue

        if response is None or response.status_code != 200:
            logger.error(
                f"Failed to download from {url} (normalized to {normalized_url})"
            )
            return None

        # Determine file type based on content or URL
        if ".ipynb" not in url and not url.endswith(".ipynb"):
            # Check if response content looks like JSON (for notebook files)
            try:
                content_str = (
                    response.text
                    if isinstance(response.text, str)
                    else response.text.encode("utf-8").decode("utf-8")
                )
                json.loads(
                    content_str[:500]
                )  # Check first 500 chars for JSON structure
                # If it's valid JSON, assume it's a notebook
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".ipynb", delete=False, encoding="utf-8"
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If not JSON, it might be a Python file
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, encoding="utf-8"
                )
        else:
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".ipynb", delete=False, encoding="utf-8"
            )

        # Write content
        if isinstance(response.content, bytes):
            temp_file.write(response.content.decode("utf-8"))
        else:
            temp_file.write(response.text)
        temp_file.close()

        logger.info(
            f"Downloaded notebook from {url} (normalized to {normalized_url}) to {temp_file.name}"
        )
        return temp_file.name
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        # Try alternative download method if regular download fails
        try:
            # For some URLs that require special handling
            if "drive.google.com" in url or "docs.google.com" in url:
                return download_google_drive_file(url)
        except:
            pass  # Ignore errors from alternative download method
        return None


def get_popular_kernels():
    """Get popular kernels from Kaggle competitions and datasets."""
    try:
        # Method 1: Try to get top kernels from competitions
        popular_kernels = []

        # Try getting trending kernels
        try:
            cmd = ["kernels", "list", "--sort-by", "hotness"]
            result = kaggle_cli_cmd(cmd)
            trending_kernels = _parse_kernels_output(result.stdout)
            popular_kernels.extend(trending_kernels[:10])  # Add top 10 trending
        except Exception:
            logger.info("Could not fetch trending kernels")

        # Try getting top kernels by votes
        try:
            cmd = ["kernels", "list", "--sort-by", "votes"]
            result = kaggle_cli_cmd(cmd)
            top_voted_kernels = _parse_kernels_output(result.stdout)
            popular_kernels.extend(top_voted_kernels[:10])  # Add top 10 voted
        except Exception:
            logger.info("Could not fetch top voted kernels")

        # Remove duplicates while preserving order
        seen = set()
        unique_kernels = []
        for kernel in popular_kernels:
            ref = kernel.get("ref", "")
            if ref not in seen:
                seen.add(ref)
                unique_kernels.append(kernel)

        return unique_kernels
    except Exception as e:
        logger.error(f"Failed to get popular kernels: {e}")
        return []


def interactive_kernel_selection(use_popular=False):
    """Interactive kernel selection with option to show popular kernels."""
    try:
        if use_popular:
            kernels = get_popular_kernels()
            if not kernels:
                logger.info("No popular kernels found, falling back to user kernels")
                kernels = kernels_list()
        else:
            kernels = kernels_list()

        if not kernels:
            logger.info("No kernels found")
            return None

        print(f"\nAvailable {'popular' if use_popular else 'your'} kernels:")
        for i, kernel in enumerate(kernels[:20], 1):  # Show up to 20 kernels
            title = kernel.get("title", "No title")
            ref = kernel.get("ref", "")
            last_run = kernel.get("lastRunTime", "N/A")
            print(f"{i:2d}. {ref:35} {title[:50]:<50} (Last run: {last_run})")

        while True:
            choice = input(
                f"\nChoose kernel number (1-{min(len(kernels), 20)}) or 'p' for popular, 'q' to quit: "
            ).strip()
            if choice.lower() == "q":
                return None
            elif choice.lower() == "p" and not use_popular:
                # Show popular kernels instead
                return interactive_kernel_selection(use_popular=True)
            elif choice.lower() == "p" and use_popular:
                # Show user's kernels instead
                return interactive_kernel_selection(use_popular=False)

            try:
                idx = int(choice) - 1
                if 0 <= idx < min(len(kernels), 20):
                    return kernels[idx]["ref"]
                print(
                    "Invalid selection, please choose a number between 1 and",
                    min(len(kernels), 20),
                )
            except ValueError:
                print("Please enter a number, 'p' for popular kernels, or 'q' to quit")

    except Exception as e:
        logger.error(f"Selection failed: {e}")
        return None


# ----------------------------
# Endpoint registry and HTTP call helpers
# ----------------------------

_REGISTRY_DIR = Path(".kaggle_state")
_REGISTRY_FILE = _REGISTRY_DIR / "endpoints.json"


def _load_endpoints() -> dict:
    try:
        if _REGISTRY_FILE.exists():
            return json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_endpoints(db: dict):
    try:
        _REGISTRY_DIR.mkdir(exist_ok=True)
        _REGISTRY_FILE.write_text(json.dumps(db, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not save endpoints registry: {e}")


def register_endpoint(name: str, url: str, provider: str, metadata: dict | None = None):
    if not name:
        return
    db = _load_endpoints()
    db[name] = {
        "url": url,
        "provider": provider,
        "metadata": metadata or {},
        "created_at": int(time.time()),
    }
    _save_endpoints(db)


def list_endpoints():
    db = _load_endpoints()
    for k, v in db.items():
        print(f"{k}: {v.get('url')} ({v.get('provider')})")


def resolve_endpoint(name_or_url: str) -> str | None:
    if name_or_url.startswith("http://") or name_or_url.startswith("https://"):
        return name_or_url
    db = _load_endpoints()
    entry = db.get(name_or_url)
    return entry.get("url") if entry else None


def http_call(url: str, method: str = "POST", json_payload: dict | None = None):
    method = method.upper()
    if method == "GET":
        return requests.get(url, timeout=60)
    return requests.post(url, json=json_payload, timeout=60)


# ----------------------------
# Command Functions
# ----------------------------


def cmd_list(args):
    """List kernels command."""
    try:
        kernels = kernels_list(args.user if hasattr(args, "user") else None)

        if not kernels:
            print("No kernels found")
            return

        print(f"{'REFERENCE':<40} {'TITLE':<50} {'LAST RUN':<25}")
        print("-" * 115)

        for kernel in kernels:
            ref = kernel.get("ref", "")[:38]
            title = kernel.get("title", "")[:48]
            last_run = kernel.get("lastRunTime", "N/A")[:23]
            print(f"{ref:<40} {title:<50} {last_run:<25}")

    except Exception as e:
        logger.error(f"Failed to list kernels: {e}")


def cmd_pull(args):
    """Pull kernel command."""
    ref = args.kernel
    dest = getattr(args, "dest", ".")

    input_type = determine_input_type(ref)

    if input_type == "kaggle_url":
        ref = parse_kaggle_url(ref)
        if not ref:
            logger.error(f"Could not parse Kaggle URL: {args.kernel}")
            return

    try:
        kernel_pull(ref, dest)
    except Exception as e:
        logger.error(f"Pull failed: {e}")


def cmd_push(args):
    """Push kernel command."""
    # Handle both 'path' and 'input' attributes for flexibility
    path = getattr(args, "path", None) or getattr(args, "input", None)

    if not path:
        logger.error("No path specified")
        return

    if not Path(path).exists():
        logger.error(f"Path not found: {path}")
        return

    try:
        kernel_push(path)
    except Exception as e:
        logger.error(f"Push failed: {e}")


def cmd_create(args):
    """Create new kernel command."""
    if hasattr(args, "input") and args.input:
        input_type = determine_input_type(args.input)

        if input_type in ["local_file", "local_dir"]:
            args.path = args.input
            return cmd_push(args)
        elif input_type == "remote_notebook":
            if temp_file := download_remote_notebook(args.input):
                args.path = temp_file
                try:
                    return cmd_push(args)
                finally:
                    os.unlink(temp_file)
            return

    # Create blank notebook
    create_blank_notebook()


def _push_and_run_from_source(source_path, timeout_sec=1800):
    """Push a local/temporary notebook (or directory) to Kaggle and wait for completion. Returns the kernel ref."""
    ref = kernel_push(source_path)
    run_notebook_in_kaggle(ref, timeout_sec=timeout_sec)
    return ref


def _download_outputs(ref, out_dir):
    """Download kernel outputs using CLI. Returns output path if available."""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        kaggle_cli_cmd(["kernels", "output", ref, "-p", str(out_dir)])
        return out_dir
    except Exception as e:
        logger.warning(f"Failed to download outputs for {ref}: {e}")
        return None


def cmd_run(args):
    """Run/execute kernel command end-to-end and optionally deploy."""
    source_dir_for_deploy = None
    deploy_opts = (
        _collect_gcp_opts_from_args(args) if getattr(args, "deploy", False) else None
    )

    if hasattr(args, "input") and args.input:
        input_type = determine_input_type(args.input)

        if input_type == "kaggle_url":
            ref = parse_kaggle_url(args.input)
            if not ref:
                logger.error(f"Could not parse Kaggle URL: {args.input}")
                return
            # Pull with metadata and then push to trigger a run
            pulled_dir = kernel_pull(ref, getattr(args, "dest", "."))
            ref = _push_and_run_from_source(pulled_dir)
            source_dir_for_deploy = pulled_dir
        elif input_type == "kaggle_ref":
            ref = args.input
            pulled_dir = kernel_pull(ref, getattr(args, "dest", "."))
            # Use pulled directory's metadata to re-push and run
            ref = _push_and_run_from_source(pulled_dir)
            source_dir_for_deploy = pulled_dir
        elif input_type in ["local_file", "local_dir"]:
            ref = _push_and_run_from_source(args.input)
            source_dir_for_deploy = (
                Path(args.input).parent
                if Path(args.input).is_file()
                else Path(args.input)
            )
        elif input_type == "remote_notebook":
            if temp_file := download_remote_notebook(args.input):
                try:
                    ref = _push_and_run_from_source(temp_file)
                    source_dir_for_deploy = Path(temp_file).parent
                finally:
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass
            else:
                return
        else:
            logger.error(f"Invalid input: {args.input}")
            return
    else:
        # Interactive selection
        ref = interactive_kernel_selection()
        if not ref:
            return
        pulled_dir = kernel_pull(ref, getattr(args, "dest", "."))
        ref = _push_and_run_from_source(pulled_dir)
        source_dir_for_deploy = pulled_dir

    # Optionally deploy
    try:
        if getattr(args, "deploy", False):
            deploy_to_gcloud(
                source_dir_for_deploy or Path(getattr(args, "dest", ".")),
                opts=deploy_opts,
            )
    except Exception as e:
        logger.error(f"Deployment step failed: {e}")


def create_blank_notebook():
    """Create a new blank notebook."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# New Kaggle Notebook\n\nCreated with runna.py CLI tool"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Your code here\nimport pandas as pd\nimport numpy as np\n\nprint('Hello Kaggle!')"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.7.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        notebook_path = temp_path / "notebook.ipynb"

        with open(notebook_path, "w") as f:
            json.dump(notebook_content, f, indent=2)

        try:
            kernel_push(temp_path)
            logger.info("Blank notebook created successfully")
        except Exception as e:
            logger.error(f"Failed to create notebook: {e}")


# ----------------------------
# Deployment Functions
# ----------------------------


def _zip_dir(source_dir: Path, out_zip: Path) -> Path:
    try:
        import zipfile

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(source_dir):
                for name in files:
                    abs_path = Path(root) / name
                    rel_path = abs_path.relative_to(source_dir)
                    zf.write(abs_path, rel_path)
        return out_zip
    except Exception as e:
        logger.error(f"Zipping failed: {e}")
        raise


def create_lambda_handler_main(lambda_py: Path, notebook_script: str):
    """Create AWS Lambda handler file from notebook code."""
    template = '''"""
AWS Lambda handler generated from Kaggle notebook.
"""
import json
import logging

# Optional model import if available
try:
    import deploy_model as _deploy_model
    _HAS_MODEL = True
except Exception:
    _deploy_model = None
    _HAS_MODEL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Notebook code
{notebook_code}

def handler(event, context):
    try:
        # event is expected to carry JSON body in API Gateway v2 format
        body = event.get('body') if isinstance(event, dict) else None
        if body and isinstance(body, str):
            try:
                payload = json.loads(body)
            except Exception:
                payload = {{'raw': body}}
        elif isinstance(event, dict):
            payload = event
        else:
            payload = {{}}

        if 'process_request' in globals() and callable(globals()['process_request']):
            result = globals()['process_request'](payload)
        elif _HAS_MODEL and hasattr(_deploy_model, 'predict'):
            features = payload.get('features') if isinstance(payload, dict) else None
            if features is None:
                return {{'statusCode': 400, 'body': json.dumps({{'error': 'Missing "features"'}})}}
            result = {{'prediction': _deploy_model.predict(features)}}
        else:
            result = {{'echo': payload}}

        return {{
            'statusCode': 200,
            'headers': {{'Content-Type': 'application/json'}},
            'body': json.dumps(result)
        }}
    except Exception as e:
        logger.exception('Lambda handler error')
        return {{'statusCode': 500, 'body': json.dumps({{'error': str(e)}})}}
'''
    lambda_py.write_text(
        template.format(notebook_code=notebook_script), encoding="utf-8"
    )
    logger.info(f"Created Lambda handler: {lambda_py}")


def package_for_aws(notebook_dir: Path) -> Path:
    """Package notebook as an AWS Lambda zip under notebook_dir/aws-lambda/.
    Returns the path to the created zip file.
    """
    notebook_dir = Path(notebook_dir)
    # Find notebook
    nb_files = list(notebook_dir.glob("*.ipynb")) or list(notebook_dir.rglob("*.ipynb"))
    if not nb_files:
        raise RuntimeError("No notebook file found for AWS packaging")
    nb = nb_files[0]

    # Convert notebook
    nb_script = convert_notebook_to_script(nb)

    # Build dir
    out_dir = notebook_dir / "aws-lambda"
    out_dir.mkdir(exist_ok=True)

    # Write handler
    lambda_file = out_dir / "lambda_function.py"
    create_lambda_handler_main(lambda_file, nb_script)

    # Include helper
    for helper in ["deploy_model.py"]:
        for candidate in [notebook_dir / helper, Path.cwd() / helper]:
            if candidate.exists():
                shutil.copy(candidate, out_dir / helper)
                logger.info(f"Included helper: {helper}")
                break

    # Requirements for Lambda (note: building deps into the zip is not handled here)
    req = out_dir / "requirements.txt"
    create_requirements_file(req, project_root=notebook_dir)

    # Zip it
    zip_path = notebook_dir / "aws-lambda.zip"
    _zip_dir(out_dir, zip_path)
    logger.info(f"AWS Lambda package created at: {zip_path}")
    return zip_path


def package_for_gcf(notebook_dir: Path) -> Path:
    """Package notebook for Google Cloud Functions (no deploy). Returns deploy dir path."""
    notebook_dir = Path(notebook_dir)
    nb_files = list(notebook_dir.glob("*.ipynb")) or list(notebook_dir.rglob("*.ipynb"))
    if not nb_files:
        raise RuntimeError("No notebook file found for GCF packaging")
    nb = nb_files[0]

    # Convert notebook
    nb_script = convert_notebook_to_script(nb)

    # Build dir
    deploy_dir = notebook_dir / "deploy"
    deploy_dir.mkdir(exist_ok=True)

    # Write main
    main_py = deploy_dir / "main.py"
    create_cloud_function_main(main_py, nb_script)

    # Include helper
    for helper in ["deploy_model.py"]:
        for candidate in [notebook_dir / helper, Path.cwd() / helper]:
            if candidate.exists():
                shutil.copy(candidate, deploy_dir / helper)
                logger.info(f"Included helper module: {helper}")
                break

    # Requirements for GCF
    req = deploy_dir / "requirements.txt"
    create_requirements_file(req, project_root=notebook_dir)

    return deploy_dir


def deploy_to_gcloud(notebook_dir, opts: dict | None = None):
    """Deploy notebook to Google Cloud Functions."""
    notebook_dir = Path(notebook_dir)

    # Find the notebook file (search recursively as needed)
    notebook_files = list(notebook_dir.glob("*.ipynb"))
    if not notebook_files:
        notebook_files = list(notebook_dir.rglob("*.ipynb"))
    if not notebook_files:
        logger.error("No notebook file found for deployment")
        return False

    notebook_file = notebook_files[0]
    logger.info(f"Deploying notebook: {notebook_file}")

    try:
        # Package only (no deploy)
        deploy_dir = package_for_gcf(notebook_dir)

        # Deploy using gcloud
        ok, fname, project, region, url = deploy_with_gcloud(deploy_dir, opts=opts)
        if ok and url:
            try:
                name_for_registry = opts.get("save_name") if opts else None
                register_endpoint(
                    name_for_registry or fname,
                    url,
                    provider="gcp-functions",
                    metadata={"project": project, "region": region},
                )
            except Exception as reg_e:
                logger.warning(f"Could not register endpoint locally: {reg_e}")
        elif ok and not url:
            logger.warning(
                "Deployment reported success, but no URL was parsed from output."
            )
        return ok

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False


def clean_jupyter_magic_commands(script_content):
    """Remove Jupyter magic commands and IPython-specific syntax from converted script.

    This function filters out:
    - Cell magic commands (%%writefile, %%time, etc.)
    - Line magic commands (%matplotlib, %load_ext, etc.)
    - get_ipython() calls
    - IPython-style comment markers (# In[...])
    """
    lines = script_content.split("\n")
    cleaned_lines = []
    skip_cell_magic = False

    for line in lines:
        stripped = line.strip()

        # Skip empty comment-only lines from nbconvert (e.g., "# In[ ]:")
        if re.match(r"^#\s*In\s*\[.*\]\s*:?\s*$", stripped):
            continue

        # Detect cell magic commands (%%command)
        if stripped.startswith("%%"):
            # Cell magic affects the entire cell, so we skip this line
            # Note: some cell magics like %%writefile write content, we skip those entirely
            logger.debug(f"Skipping cell magic: {stripped}")
            continue

        # Detect line magic commands (%command)
        if stripped.startswith("%") and not stripped.startswith("%%"):
            logger.debug(f"Skipping line magic: {stripped}")
            continue

        # Skip get_ipython() calls (often used by nbconvert for magic translation)
        if "get_ipython()" in line:
            logger.debug(f"Skipping get_ipython() call: {stripped}")
            continue

        # Keep the line
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)

    # Remove excessive blank lines (more than 2 consecutive)
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    return result


def convert_notebook_to_script(notebook_file):
    """Convert Jupyter notebook to Python script using nbconvert when available."""
    try:
        try:
            import nbformat
            from nbconvert import PythonExporter

            with open(notebook_file, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            exporter = PythonExporter()
            body, _ = exporter.from_notebook_node(nb)
            # Clean Jupyter magic commands and IPython syntax
            body = clean_jupyter_magic_commands(body)
            return body
        except Exception as e:
            logger.warning(f"nbconvert failed ({e}), falling back to simple extraction")
            with open(notebook_file, "r", encoding="utf-8") as f:
                notebook = json.load(f)
            script_lines = []
            script_lines.append("# Converted from Jupyter notebook")
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", [])
                    if isinstance(source, list):
                        for line in source:
                            # Skip magic commands in fallback mode too
                            stripped = line.strip()
                            if (
                                not stripped.startswith("%")
                                and "get_ipython()" not in line
                            ):
                                script_lines.append(line)
                    else:
                        stripped = source.strip()
                        if (
                            not stripped.startswith("%")
                            and "get_ipython()" not in source
                        ):
                            script_lines.append(source)
                    script_lines.append("")
            result = "\n".join(script_lines)
            result = clean_jupyter_magic_commands(result)
            return result
    except Exception as e:
        logger.error(f"Failed to convert notebook: {e}")
        raise


def create_cloud_function_main(main_py, notebook_script):
    """Create main.py for Google Cloud Functions."""
    template = '''"""
Google Cloud Function generated from Kaggle notebook.
"""
import json
import logging
from flask import Flask, request, jsonify

# Optional model import if available
try:
    import deploy_model as _deploy_model
    _HAS_MODEL = True
except Exception:
    _deploy_model = None
    _HAS_MODEL = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Notebook code
{notebook_code}

def predict_handler(request):
    """HTTP Cloud Function entry point."""
    try:
        # Handle CORS
        if request.method == 'OPTIONS':
            headers = {{
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Max-Age': '3600'
            }}
            return ('', 204, headers)

        # Set CORS headers for main request
        headers = {{'Access-Control-Allow-Origin': '*'}}

        if request.method == 'POST':
            request_json = request.get_json(silent=True)
            if request_json is None:
                return jsonify({{'error': 'No JSON data provided'}}), 400, headers
            # Use notebook-defined processing if present; otherwise call deploy_model.predict if available; else echo
            try:
                if 'process_request' in globals() and callable(globals()['process_request']):
                    result = globals()['process_request'](request_json)
                elif _HAS_MODEL and hasattr(_deploy_model, 'predict'):
                    # Minimal example: expects features in JSON under "features"
                    features = request_json.get('features')
                    if features is None:
                        return jsonify({{'error': 'Missing "features" in request body'}}), 400, headers
                    # This is a placeholder; real model would require proper loading
                    result = {{'prediction': _deploy_model.predict(features)}}
                else:
                    result = {{'echo': request_json}}
            except Exception as inner_e:
                logger.exception('Processing error')
                return jsonify({{'error': str(inner_e)}}), 500, headers
            return jsonify(result), 200, headers
        elif request.method == 'GET':
            return jsonify({{'status': 'Kaggle notebook API is running'}}), 200, headers
        else:
            return jsonify({{'error': 'Method not allowed'}}), 405, headers
    except Exception as e:
        logger.error(f"Error processing request: {{e}}")
        return jsonify({{'error': str(e)}}), 500, headers

# For local testing
if __name__ == '__main__':
    app = Flask(__name__)
    app.add_url_rule('/', 'predict_handler', predict_handler, methods=['GET', 'POST'])
    app.run(debug=True, port=8080)
'''

    with open(main_py, "w") as f:
        f.write(template.format(notebook_code=notebook_script))

    logger.info(f"Created Cloud Function main.py: {main_py}")


def _detect_model_requirements(project_root: Path) -> list:
    """Scan deploy_model.py to determine extra requirements for deployment."""
    extras = []
    try:
        candidates = [project_root / "deploy_model.py", Path.cwd() / "deploy_model.py"]
        dm = None
        for c in candidates:
            if c.exists():
                dm = c
                break
        if not dm:
            return extras
        text = dm.read_text(encoding="utf-8", errors="ignore")

        def seen(pkg):
            if pkg not in extras:
                extras.append(pkg)

        if re.search(r"\bimport\s+torch\b|from\s+torch\s+import\b", text):
            seen("torch")
        if re.search(r"\bimport\s+sklearn\b|from\s+sklearn\s+import\b", text):
            seen("scikit-learn")
        if re.search(r"\bimport\s+numpy\b|from\s+numpy\s+import\b", text):
            seen("numpy")
        if re.search(r"\bimport\s+pandas\b|from\s+pandas\s+import\b", text):
            seen("pandas")
        if re.search(r"\bimport\s+requests\b|from\s+requests\s+import\b", text):
            seen("requests")
        if re.search(r"\bimport\s+kagglehub\b|from\s+kagglehub\s+import\b", text):
            seen("kagglehub")
        if re.search(
            r"\bimport\s+huggingface_hub\b|from\s+huggingface_hub\s+import\b", text
        ):
            seen("huggingface_hub")
        if re.search(r"\bimport\s+dotenv\b|from\s+dotenv\s+import\b", text):
            seen("python-dotenv")
    except Exception as e:
        logger.debug(f"model requirements detection failed: {e}")
    return extras


def create_requirements_file(requirements_txt, project_root: Path):
    """Create requirements.txt for Cloud Functions, adding model deps if present."""
    requirements = ["flask>=2.0.0", "requests>=2.25.0", "functions-framework>=3.5.0"]
    for pkg in _detect_model_requirements(project_root):
        if pkg not in requirements:
            requirements.append(pkg)

    with open(requirements_txt, "w") as f:
        f.write("\n".join(requirements))

    logger.info(f"Created requirements.txt: {requirements_txt}")


def _prompt_if_missing(val, prompt_text):
    if val:
        return val
    try:
        return input(prompt_text).strip()
    except Exception:
        return val


def _parse_function_url(text: str) -> str | None:
    try:
        # Look for a URL pattern in stdout
        m = re.search(r"https://[\w\-\.]+cloudfunctions\.net/[\w\-]+", text)
        if m:
            return m.group(0)
    except Exception:
        pass
    return None


def deploy_with_gcloud(deploy_dir: Path, opts: dict | None = None):
    """Deploy to Google Cloud Functions using gcloud CLI.
    opts keys supported: project, region, function_name, memory, timeout, public
    """
    opts = opts or {}
    try:
        # Check if gcloud is installed
        subprocess.run(["gcloud", "--version"], check=True, capture_output=True)

        project = (
            opts.get("project")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("PROJECT_ID")
        )
        region = opts.get("region") or os.environ.get("REGION", "us-central1")
        function_name = (
            opts.get("function_name") or f"kaggle-notebook-{int(time.time())}"
        )
        memory = opts.get("memory") or "512MB"
        timeout = opts.get("timeout") or "540s"
        public = bool(opts.get("public", True))

        # Prompt for project if still missing
        project = _prompt_if_missing(project, "Enter GCP Project ID: ")
        if not project:
            raise RuntimeError("GCP Project ID is required to deploy")

        cmd = [
            "gcloud",
            "functions",
            "deploy",
            function_name,
            "--project",
            project,
            "--runtime",
            "python311",
            "--trigger-http",
            "--memory",
            memory,
            "--timeout",
            timeout,
            "--region",
            region,
            "--source",
            ".",
            "--entry-point",
            "predict_handler",
        ]
        if public:
            cmd.append("--allow-unauthenticated")

        logger.info(f"Deploying to Google Cloud Functions: {function_name}")
        logger.info(f"Command: {' '.join(cmd)}")

        # Run in deploy_dir so --source '.' is correct
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, cwd=str(deploy_dir)
        )

        logger.info("Deployment successful!")
        logger.info(result.stdout)

        url = _parse_function_url(result.stdout) or ""
        if url:
            logger.info(f"Function URL: {url}")
        return True, function_name, project, region, url

    except subprocess.CalledProcessError as e:
        logger.error(f"gcloud deployment failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False, None, None, None, None
    except FileNotFoundError:
        logger.error("gcloud CLI not found. Please install Google Cloud SDK")
        return False, None, None, None, None


def _collect_gcp_opts_from_args(args) -> dict:
    return {
        "project": getattr(args, "gcp_project", None)
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("PROJECT_ID"),
        "region": getattr(args, "region", None)
        or os.environ.get("REGION", "us-central1"),
        "function_name": getattr(args, "function_name", None),
        "memory": getattr(args, "memory", None) or "512MB",
        "timeout": getattr(args, "timeout", None) or "540s",
        "public": getattr(args, "public", True),
        "save_name": getattr(args, "save_name", None),
    }


def cmd_deploy(args):
    """Deploy notebook command."""
    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path not found: {path}")
        return

    try:
        opts = _collect_gcp_opts_from_args(args)
        success = deploy_to_gcloud(path, opts=opts)
        if success:
            logger.info("Deployment completed successfully!")
        else:
            logger.error("Deployment failed")
    except Exception as e:
        logger.error(f"Deploy failed: {e}")


def cmd_package_aws(args):
    """Package notebook as AWS Lambda zip (no deployment)."""
    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path not found: {path}")
        return
    try:
        zip_file = package_for_aws(path)
        print(f"AWS Lambda package ready: {zip_file}")
        print("To deploy with AWS CLI (example):")
        print(
            "aws lambda create-function --function-name my-func --runtime python3.11 --zip-file fileb://aws-lambda.zip --handler lambda_function.handler --role arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME>"
        )
    except Exception as e:
        logger.error(f"AWS packaging failed: {e}")


def deploy_to_aws_lambda(notebook_dir, opts: dict | None = None):
    """Deploy notebook to AWS Lambda using AWS CLI."""
    notebook_dir = Path(notebook_dir)
    opts = opts or {}

    try:
        # Check if AWS CLI is installed
        subprocess.run(["aws", "--version"], check=True, capture_output=True)

        # Package first
        zip_file = package_for_aws(notebook_dir)

        function_name = (
            opts.get("function_name") or f"kaggle-notebook-{int(time.time())}"
        )
        role_arn = opts.get("role_arn") or os.environ.get("AWS_LAMBDA_ROLE_ARN")
        region = opts.get("region") or os.environ.get("AWS_REGION", "us-east-1")
        memory = opts.get("memory") or "512"
        timeout = opts.get("timeout") or "300"

        # Prompt for role ARN if missing
        if not role_arn:
            role_arn = input(
                "Enter AWS Lambda Role ARN (arn:aws:iam::ACCOUNT:role/ROLE_NAME): "
            ).strip()

        if not role_arn:
            logger.error("AWS Lambda Role ARN is required for deployment")
            return False, None, None

        # Check if function exists
        check_cmd = [
            "aws",
            "lambda",
            "get-function",
            "--function-name",
            function_name,
            "--region",
            region,
        ]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        function_exists = result.returncode == 0

        if function_exists:
            # Update existing function
            logger.info(f"Updating existing Lambda function: {function_name}")
            cmd = [
                "aws",
                "lambda",
                "update-function-code",
                "--function-name",
                function_name,
                "--zip-file",
                f"fileb://{zip_file}",
                "--region",
                region,
            ]
        else:
            # Create new function
            logger.info(f"Creating new Lambda function: {function_name}")
            cmd = [
                "aws",
                "lambda",
                "create-function",
                "--function-name",
                function_name,
                "--runtime",
                "python3.11",
                "--role",
                role_arn,
                "--handler",
                "lambda_function.handler",
                "--zip-file",
                f"fileb://{zip_file}",
                "--timeout",
                str(timeout),
                "--memory-size",
                str(memory),
                "--region",
                region,
            ]

        logger.info(f"Deploying to AWS Lambda: {function_name}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("AWS Lambda deployment successful!")

        # Try to get function URL if configured
        url_cmd = [
            "aws",
            "lambda",
            "get-function-url-config",
            "--function-name",
            function_name,
            "--region",
            region,
        ]
        url_result = subprocess.run(url_cmd, capture_output=True, text=True)
        url = None
        if url_result.returncode == 0:
            import json

            url_data = json.loads(url_result.stdout)
            url = url_data.get("FunctionUrl")

        if url:
            logger.info(f"Function URL: {url}")
        else:
            logger.info(
                f"Function deployed. To create a URL, run:\naws lambda create-function-url-config --function-name {function_name} --auth-type NONE --region {region}"
            )

        return True, function_name, url

    except subprocess.CalledProcessError as e:
        logger.error(f"AWS Lambda deployment failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False, None, None
    except FileNotFoundError:
        logger.error(
            "AWS CLI not found. Please install AWS CLI: https://aws.amazon.com/cli/"
        )
        return False, None, None
    except Exception as e:
        logger.error(f"AWS Lambda deployment error: {e}")
        return False, None, None


def package_for_modal(notebook_dir: Path, opts: dict = None) -> Path:
    """Package notebook for Modal.com deployment. Returns deploy dir path."""
    notebook_dir = Path(notebook_dir)
    opts = opts or {}
    nb_files = list(notebook_dir.glob("*.ipynb")) or list(notebook_dir.rglob("*.ipynb"))
    if not nb_files:
        raise RuntimeError("No notebook file found for Modal packaging")
    nb = nb_files[0]

    # Convert notebook
    nb_script = convert_notebook_to_script(nb)

    # Build dir
    deploy_dir = notebook_dir / "modal-deploy"
    deploy_dir.mkdir(exist_ok=True)

    # Create Modal app file
    modal_app = deploy_dir / "modal_app.py"
    create_modal_app(modal_app, nb_script, opts)

    # Include helper
    for helper in ["deploy_model.py"]:
        for candidate in [notebook_dir / helper, Path.cwd() / helper]:
            if candidate.exists():
                shutil.copy(candidate, deploy_dir / helper)
                logger.info(f"Included helper module: {helper}")
                break

    # Requirements for Modal
    req = deploy_dir / "requirements.txt"
    create_requirements_file(req, project_root=notebook_dir)

    return deploy_dir


def create_modal_app(modal_py: Path, notebook_script: str, opts: dict = None):
    """Create Modal.com app file from notebook code with GPU/secrets support."""
    opts = opts or {}
    gpu = opts.get("gpu", "None")
    secrets = opts.get("secrets", [])
    timeout = opts.get("timeout", 300)
    
    secrets_str = f"[{', '.join([f'modal.Secret.from_name(\"{s}\")' for s in secrets])}]" if secrets else "[]"
    
    template = '''"""
Modal.com app generated from Kaggle notebook.
Install: pip install modal
Deploy: modal deploy modal_app.py
Run: modal run modal_app.py
"""
import modal

app = modal.App("kaggle-notebook")

image = modal.Image.debian_slim().pip_install(
    "numpy", "pandas", "scikit-learn", "requests"
)

volume = modal.Volume.from_name("model-cache", create_if_missing=True)

try:
    import deploy_model as _deploy_model
    _HAS_MODEL = True
except Exception:
    _deploy_model = None
    _HAS_MODEL = False

{notebook_code}

@app.function(
    image=image,
    gpu={gpu},
    secrets={secrets},
    volumes={{"/cache": volume}},
    timeout={timeout},
)
@modal.web_endpoint(method="POST")
def predict(data: dict):
    """Modal web endpoint for predictions."""
    try:
        if 'process_request' in globals() and callable(globals()['process_request']):
            result = globals()['process_request'](data)
        elif _HAS_MODEL and hasattr(_deploy_model, 'predict'):
            features = data.get('features')
            if features is None:
                return {{"error": "Missing 'features' in request body"}}, 400
            result = {{"prediction": _deploy_model.predict(features)}}
        else:
            result = {{"echo": data}}
        return result
    except Exception as e:
        return {{"error": str(e)}}, 500

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {{"status": "ok", "message": "Kaggle notebook API is running"}}

@app.local_entrypoint()
def main():
    print("Deploy: modal deploy modal_app.py")
    print("Serve: modal serve modal_app.py")
'''

    modal_py.write_text(
        template.format(
            notebook_code=notebook_script,
            gpu=gpu,
            secrets=secrets_str,
            timeout=timeout
        ), encoding="utf-8"
    )
    logger.info(f"Created Modal app: {modal_py}")


def deploy_to_modal(notebook_dir, opts: dict | None = None):
    """Deploy notebook to Modal.com with GPU/secrets support."""
    notebook_dir = Path(notebook_dir)
    opts = opts or {}

    try:
        # Check if modal is installed
        try:
            import modal

            logger.info("Modal SDK found")
        except ImportError:
            logger.error("Modal SDK not found. Install with: pip install modal")
            return False, None

        # Package for Modal
        deploy_dir = package_for_modal(notebook_dir, opts)
        modal_app = deploy_dir / "modal_app.py"

        if not modal_app.exists():
            logger.error("modal_app.py not found")
            return False, None

        # Check if user is authenticated
        try:
            subprocess.run(
                ["modal", "token", "set", "--help"], check=True, capture_output=True
            )
        except:
            logger.warning("Modal authentication may be required. Run: modal token new")

        # Deploy using modal CLI
        logger.info("Deploying to Modal.com...")
        cmd = ["modal", "deploy", str(modal_app)]
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, cwd=str(deploy_dir)
        )

        logger.info("Modal deployment successful!")
        logger.info(result.stdout)

        # Try to extract URL from output
        url = None
        for line in result.stdout.split("\n"):
            if "https://" in line and "modal.run" in line:
                # Extract URL from line
                import re

                urls = re.findall(r"https://[^\s]+", line)
                if urls:
                    url = urls[0]
                    break

        if url:
            logger.info(f"Modal endpoint URL: {url}")

        return True, url

    except subprocess.CalledProcessError as e:
        logger.error(f"Modal deployment failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False, None
    except Exception as e:
        logger.error(f"Modal deployment error: {e}")
        return False, None


def cmd_deploy_aws(args):
    """Deploy notebook to AWS Lambda command."""
    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path not found: {path}")
        return

    try:
        opts = {
            "function_name": getattr(args, "function_name", None),
            "role_arn": getattr(args, "role_arn", None),
            "region": getattr(args, "region", None),
            "memory": getattr(args, "memory", None),
            "timeout": getattr(args, "timeout", None),
        }
        success, function_name, url = deploy_to_aws_lambda(path, opts=opts)
        if success:
            logger.info(f"AWS Lambda deployment completed: {function_name}")
            if url and getattr(args, "save_name", None):
                try:
                    register_endpoint(
                        args.save_name,
                        url,
                        provider="aws-lambda",
                        metadata={
                            "function": function_name,
                            "region": opts.get("region", "us-east-1"),
                        },
                    )
                except Exception as reg_e:
                    logger.warning(f"Could not register endpoint: {reg_e}")
        else:
            logger.error("AWS Lambda deployment failed")
    except Exception as e:
        logger.error(f"Deploy to AWS failed: {e}")


def cmd_deploy_modal(args):
    """Deploy notebook to Modal.com command with GPU/secrets support."""
    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path not found: {path}")
        return

    try:
        opts = {
            "gpu": getattr(args, "gpu", None),
            "secrets": getattr(args, "secrets", []),
            "timeout": getattr(args, "timeout", 300),
        }
        success, url = deploy_to_modal(path, opts)
        if success:
            logger.info("Modal.com deployment completed!")
            if url and getattr(args, "save_name", None):
                try:
                    register_endpoint(
                        args.save_name, url, provider="modal", metadata=opts
                    )
                except Exception as reg_e:
                    logger.warning(f"Could not register endpoint: {reg_e}")
        else:
            logger.error("Modal deployment failed")
    except Exception as e:
        logger.error(f"Deploy to Modal failed: {e}")


def cmd_app_list(args):
    """List apps in library."""
    from app_library import list_apps
    apps = list_apps()
    if not apps:
        print("No apps in library. Use 'app-add' to add apps.")
        return
    
    print(f"\n{'Name':<20} {'GPU':<10} {'Deployed':<15} {'Description'}")
    print("-" * 80)
    for name, info in apps.items():
        gpu = info.get('gpu', 'None')
        deployed = '✓' if info.get('deployed') else '✗'
        desc = info.get('description', '')[:40]
        print(f"{name:<20} {gpu:<10} {deployed:<15} {desc}")


def cmd_app_show(args):
    """Show app code."""
    from app_library import get_app, list_apps
    code = get_app(args.name)
    if not code:
        logger.error(f"App not found: {args.name}")
        return
    
    apps = list_apps()
    info = apps.get(args.name, {})
    print(f"\n# {args.name}")
    print(f"# {info.get('description', '')}")
    print(f"# GPU: {info.get('gpu', 'None')}")
    print(f"# Tags: {', '.join(info.get('tags', []))}\n")
    print(code)


def cmd_app_deploy(args):
    """Deploy app from library."""
    from app_library import get_app, update_app, list_apps
    
    code = get_app(args.name)
    if not code:
        logger.error(f"App not found: {args.name}")
        return
    
    # Write to temp file
    temp_dir = Path(f"/tmp/app_{args.name}")
    temp_dir.mkdir(exist_ok=True)
    app_file = temp_dir / "app.py"
    app_file.write_text(code)
    
    # Get app info for GPU setting
    apps = list_apps()
    info = apps.get(args.name, {})
    gpu = getattr(args, "gpu", None) or info.get("gpu")
    
    # Deploy
    opts = {
        "gpu": gpu,
        "secrets": getattr(args, "secrets", []),
        "timeout": getattr(args, "timeout", 300),
    }
    
    success, url = deploy_to_modal(temp_dir, opts)
    if success and url:
        update_app(args.name, deployed_url=url)
        logger.info(f"App deployed: {url}")
    else:
        logger.error("Deployment failed")


def cmd_app_add(args):
    """Add app to library."""
    from app_library import add_app
    
    if args.file:
        code = Path(args.file).read_text()
    else:
        logger.error("Must provide --file")
        return
    
    add_app(
        args.name,
        code,
        description=getattr(args, "description", ""),
        tags=getattr(args, "tags", []),
        gpu=getattr(args, "gpu", None)
    )
    logger.info(f"Added app: {args.name}")


def cmd_app_update(args):
    """Update app in library."""
    from app_library import update_app
    
    if args.file:
        code = Path(args.file).read_text()
        update_app(args.name, code=code)
        logger.info(f"Updated app: {args.name}")
    else:
        logger.error("Must provide --file")


def cmd_app_delete(args):
    """Delete app from library."""
    from app_library import delete_app
    
    if delete_app(args.name):
        logger.info(f"Deleted app: {args.name}")
    else:
        logger.error(f"App not found: {args.name}")


def cmd_serve_local(args):
    """Package notebook for local serving and optionally run functions-framework."""
    path = Path(args.path)
    port = int(getattr(args, "port", 8080))
    if not path.exists():
        logger.error(f"Path not found: {path}")
        return
    try:
        # Ensure deploy dir exists and main.py generated (no deploy)
        deploy_dir = package_for_gcf(path)
        if not (deploy_dir / "main.py").exists():
            logger.error("deploy/main.py not found; cannot serve locally")
            return
        main_py = deploy_dir / "main.py"
        if not main_py.exists():
            logger.error("deploy/main.py not found; cannot serve locally")
            return
        # Optionally run functions-framework
        if getattr(args, "run", False):
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "functions_framework",
                    "--target",
                    "predict_handler",
                    "--port",
                    str(port),
                ]
                logger.info(f"Starting local server: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, cwd=str(deploy_dir))
            except Exception as e:
                logger.error(f"Local serve failed: {e}")
        else:
            print(f"Local package ready at {deploy_dir}. To run:")
            print(
                f"(cd {deploy_dir} && {sys.executable} -m functions_framework --target predict_handler --port {port})"
            )
    except Exception as e:
        logger.error(f"Local packaging/serve failed: {e}")


# ----------------------------
# Main CLI
# ----------------------------


def safe_input(prompt, default=None, input_type=str, validation_func=None):
    """Safely get input from user with validation and default values."""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default

            if input_type == int:
                value = int(user_input)
            elif input_type == bool:
                value = user_input.lower() in ["true", "yes", "1", "y"]
            else:
                value = input_type(user_input)

            if validation_func:
                if validation_func(value):
                    return value
                else:
                    print("Invalid input. Please try again.")
                    continue
            else:
                return value
        except ValueError:
            print(
                f"Invalid input type. Expected {input_type.__name__}. Please try again."
            )
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")


def validate_setup():
    """Validate that required tools are available."""
    issues = []

    # Check for kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("Kaggle CLI not found. Install with: pip install kaggle")

    # Check for gcloud CLI (optional)
    try:
        subprocess.run(["gcloud", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append(
            "gcloud CLI not found (optional for deployment). Install Google Cloud SDK"
        )

    # Check for credentials
    try:
        get_auth()
    except RuntimeError as e:
        issues.append(f"Authentication issue: {e}")

    if issues:
        logger.warning("Setup validation found issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Setup validation passed!")

    return len(issues) == 0


def cmd_batch(args):
    """Process multiple notebooks in batch."""
    try:
        if not Path(args.input_file).exists():
            logger.error(f"Input file not found: {args.input_file}")
            return

        # Read notebook list from file
        with open(args.input_file, "r", encoding="utf-8") as f:
            notebook_list = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        logger.info(f"Processing {len(notebook_list)} notebooks in batch")
        success_count = 0

        for i, notebook_ref in enumerate(notebook_list):
            logger.info(f"Processing {i+1}/{len(notebook_list)}: {notebook_ref}")
            try:
                if args.operation == "download":
                    # For download, create a temporary args object for pull command
                    class TempArgs:
                        def __init__(self, kernel, dest):
                            self.kernel = kernel
                            self.dest = dest

                    pull_args = TempArgs(kernel=notebook_ref, dest=args.output_dir)
                    cmd_pull(pull_args)
                elif args.operation == "deploy":
                    # For deploy, create a temporary args object
                    class TempDeployArgs:
                        def __init__(
                            self,
                            path,
                            gcp_project,
                            region,
                            function_name,
                            memory,
                            timeout,
                            save_name,
                            public,
                        ):
                            self.path = str(path)
                            self.gcp_project = gcp_project
                            self.region = region
                            self.function_name = function_name
                            self.memory = memory
                            self.timeout = timeout
                            self.save_name = save_name
                            self.public = public

                    deploy_args = TempDeployArgs(
                        path=notebook_ref,
                        gcp_project=None,
                        region=None,
                        function_name=None,
                        memory="512MB",
                        timeout="540s",
                        save_name=None,
                        public=True,
                    )
                    # First pull if it's a remote reference
                    if "kaggle.com" in str(notebook_ref) or "/" in str(notebook_ref):
                        temp_dir = Path(tempfile.mkdtemp())
                        pull_args = type(
                            "TempArgs",
                            (),
                            {"kernel": notebook_ref, "dest": str(temp_dir)},
                        )()
                        pulled_dir = kernel_pull(notebook_ref, str(temp_dir))
                        deploy_args.path = str(pulled_dir)
                    cmd_deploy(deploy_args)
                elif args.operation == "run":
                    # For run, create a temporary args object
                    class TempRunArgs:
                        def __init__(self, input_val, dest, deploy):
                            self.input = input_val
                            self.dest = dest
                            self.deploy = deploy

                    run_args = TempRunArgs(
                        input_val=notebook_ref, dest=args.output_dir, deploy=False
                    )
                    cmd_run(run_args)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to {args.operation} {notebook_ref}: {e}")

        logger.info(
            f"Batch processing completed: {success_count}/{len(notebook_list)} successful"
        )
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")


def cmd_preprocess(args):
    """Preprocess notebook files."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return

    try:
        if input_path.is_file() and input_path.suffix == ".ipynb":
            notebooks = [input_path]
        elif input_path.is_dir():
            notebooks = list(input_path.glob("*.ipynb"))
        else:
            logger.error(
                f"Input path must be a notebook file or directory: {input_path}"
            )
            return

        processed_count = 0
        for notebook in notebooks:
            logger.info(f"Preprocessing: {notebook}")
            processed_nb = preprocess_notebook(
                str(notebook),
                clean_metadata=args.clean_metadata,
                remove_outputs=args.remove_outputs,
                scan_security=args.scan_security,
                optimize_imports=args.optimize_imports,
            )

            if processed_nb:
                # Determine output path
                output_path = Path(args.output) if args.output else notebook
                if args.output and len(notebooks) > 1:
                    output_path = Path(args.output) / notebook.name

                output_path.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure output directory exists
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_nb, f, indent=2)
                logger.info(f"Saved processed notebook: {output_path}")
                processed_count += 1

        logger.info(
            f"Preprocessing completed: {processed_count}/{len(notebooks)} notebooks processed"
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")


def preprocess_notebook(
    notebook_path,
    clean_metadata=False,
    remove_outputs=False,
    scan_security=False,
    optimize_imports=False,
):
    """Apply preprocessing operations to notebook."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    operations_applied = []

    if clean_metadata:
        notebook = clean_notebook_metadata(notebook)
        operations_applied.append("clean_metadata")

    if remove_outputs:
        notebook = remove_notebook_outputs(notebook)
        operations_applied.append("remove_outputs")

    if scan_security:
        scan_notebook_security(notebook, notebook_path)
        operations_applied.append("scan_security")

    if optimize_imports:
        notebook = optimize_notebook_imports(notebook)
        operations_applied.append("optimize_imports")

    logger.info(f"Applied preprocessing operations: {operations_applied}")
    return notebook


def clean_notebook_metadata(notebook):
    """Remove unnecessary metadata from notebook."""
    if "metadata" in notebook:
        # Remove execution info and other unnecessary metadata
        if "kernelspec" in notebook["metadata"]:
            # Keep essential kernelspec info but clean others
            pass

    # Clean cell metadata
    for cell in notebook.get("cells", []):
        if "metadata" in cell:
            cell["metadata"] = {}  # Clear cell metadata
        if "execution_count" in cell:
            del cell["execution_count"]  # Remove execution counts

    return notebook


def remove_notebook_outputs(notebook):
    """Remove output cells from notebook."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code" and "outputs" in cell:
            cell["outputs"] = []  # Clear outputs
    return notebook


def scan_notebook_security(notebook, notebook_path):
    """Scan notebook for hardcoded credentials and security issues."""
    issues_found = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = (
                "".join(cell.get("source", []))
                if isinstance(cell.get("source"), list)
                else cell.get("source", "")
            )
            # Check for common credential patterns
            import re

            credential_patterns = [
                r"(?:API_KEY|API_SECRET|SECRET_KEY|ACCESS_KEY|TOKEN|PASSWORD|PASSWD|PWD)",
                r"(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|PWD)|\w{30,}",  # Simplified pattern',
            ]

            for pattern in credential_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    issues_found.append(
                        f"Potential credential found in cell: {source[:100]}..."
                    )

    if issues_found:
        logger.warning(
            f"Security scan found {len(issues_found)} potential issues in {notebook_path}:"
        )
        for issue in issues_found:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Security scan completed with no issues found")

    return issues_found


def optimize_notebook_imports(notebook):
    """Analyze and optimize import statements in notebook."""
    import_statements = set()

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = (
                "".join(cell.get("source", []))
                if isinstance(cell.get("source"), list)
                else cell.get("source", "")
            )
            # Find import statements
            import_lines = [
                line.strip()
                for line in source.split("\n")
                if line.strip().startswith(("import ", "from "))
            ]
            for imp in import_lines:
                import_statements.add(imp)

    logger.info(f"Found {len(import_statements)} unique import statements")
    # Could optimize by consolidating imports to early cells, etc.
    return notebook


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Kaggle CLI with deployment capabilities",
        epilog="""
Examples:
  %(prog)s list                                    # List your kernels
  %(prog)s pull username/kernel-name               # Pull a kernel
  %(prog)s run username/kernel-name --deploy       # Run and deploy to GCloud
  %(prog)s run https://kaggle.com/code/user/kernel # Run from URL
  %(prog)s create notebook.ipynb                   # Create from local file
  %(prog)s deploy ./notebook-directory             # Deploy existing notebook
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List kernels")
    list_parser.add_argument("--user", help="Username to list kernels for")
    list_parser.add_argument("--search", help="Search term for kernels")
    list_parser.set_defaults(func=cmd_list)

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull/download kernel")
    pull_parser.add_argument(
        "kernel", help="Kernel reference (user/kernel) or Kaggle URL"
    )
    pull_parser.add_argument(
        "--dest", default=".", help="Destination directory (default: current)"
    )
    pull_parser.set_defaults(func=cmd_pull)

    # Push command
    push_parser = subparsers.add_parser("push", help="Push/upload kernel")
    push_parser.add_argument("path", help="Path to notebook file or directory")
    push_parser.set_defaults(func=cmd_push)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create new kernel")
    create_parser.add_argument(
        "input", nargs="?", help="Local file, remote URL, or empty for blank notebook"
    )
    create_parser.set_defaults(func=cmd_create)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run/execute kernel")
    run_parser.add_argument(
        "input",
        nargs="?",
        help="Kernel ref, URL, local file, or empty for interactive selection",
    )
    run_parser.add_argument(
        "--dest", default=".", help="Destination directory for pulled files"
    )
    run_parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy to Google Cloud Functions after running",
    )
    # GCP deploy options for run --deploy
    run_parser.add_argument("--gcp-project", help="GCP Project ID")
    run_parser.add_argument(
        "--region", default=None, help="GCP region (default: env REGION or us-central1)"
    )
    run_parser.add_argument(
        "--function-name",
        dest="function_name",
        help="Cloud Function name (default autogenerated)",
    )
    run_parser.add_argument(
        "--memory", default="512MB", help="Function memory (default: 512MB)"
    )
    run_parser.add_argument(
        "--timeout", default="540s", help="Function timeout (default: 540s)"
    )
    run_parser.add_argument(
        "--save-name",
        dest="save_name",
        help="Save endpoint under this name for later use",
    )
    run_parser.add_argument(
        "--private",
        dest="public",
        action="store_false",
        help="Require authentication; default is public",
    )
    run_parser.set_defaults(func=cmd_run, public=True)

    # Deploy command
    deploy_parser = subparsers.add_parser(
        "deploy", help="Deploy existing notebook directory or file"
    )
    deploy_parser.add_argument(
        "path", help="Path to notebook file or directory containing notebook"
    )
    deploy_parser.add_argument("--gcp-project", help="GCP Project ID")
    deploy_parser.add_argument(
        "--region", default=None, help="GCP region (default: env REGION or us-central1)"
    )
    deploy_parser.add_argument(
        "--function-name",
        dest="function_name",
        help="Cloud Function name (default autogenerated)",
    )
    deploy_parser.add_argument(
        "--memory", default="512MB", help="Function memory (default: 512MB)"
    )
    deploy_parser.add_argument(
        "--timeout", default="540s", help="Function timeout (default: 540s)"
    )
    deploy_parser.add_argument(
        "--save-name",
        dest="save_name",
        help="Save endpoint under this name for later use",
    )
    deploy_parser.add_argument(
        "--private",
        dest="public",
        action="store_false",
        help="Require authentication; default is public",
    )
    deploy_parser.set_defaults(func=cmd_deploy, public=True)

    # AWS packaging command
    aws_pkg_parser = subparsers.add_parser(
        "package-aws", help="Package notebook as AWS Lambda zip (no deploy)"
    )
    aws_pkg_parser.add_argument(
        "path", help="Path to notebook file or directory containing notebook"
    )
    aws_pkg_parser.set_defaults(func=cmd_package_aws)

    # AWS Lambda deployment command
    aws_deploy_parser = subparsers.add_parser(
        "deploy-aws", help="Deploy notebook to AWS Lambda"
    )
    aws_deploy_parser.add_argument(
        "path", help="Path to notebook file or directory containing notebook"
    )
    aws_deploy_parser.add_argument("--function-name", help="Lambda function name")
    aws_deploy_parser.add_argument("--role-arn", help="AWS IAM role ARN for Lambda")
    aws_deploy_parser.add_argument("--region", help="AWS region (default: us-east-1)")
    aws_deploy_parser.add_argument("--memory", help="Memory size in MB (default: 512)")
    aws_deploy_parser.add_argument(
        "--timeout", help="Timeout in seconds (default: 300)"
    )
    aws_deploy_parser.add_argument(
        "--save-name", help="Save endpoint under this name for later use"
    )
    aws_deploy_parser.set_defaults(func=cmd_deploy_aws)

    # Modal.com deployment command
    modal_deploy_parser = subparsers.add_parser(
        "deploy-modal", help="Deploy notebook to Modal.com"
    )
    modal_deploy_parser.add_argument(
        "path", help="Path to notebook file or directory containing notebook"
    )
    modal_deploy_parser.add_argument(
        "--gpu", help="GPU type: T4, A10G, A100, or None (default: None)"
    )
    modal_deploy_parser.add_argument(
        "--secrets", nargs="*", default=[], help="Modal secret names to attach"
    )
    modal_deploy_parser.add_argument(
        "--timeout", type=int, default=300, help="Function timeout in seconds (default: 300)"
    )
    modal_deploy_parser.add_argument(
        "--save-name", help="Save endpoint under this name for later use"
    )
    modal_deploy_parser.set_defaults(func=cmd_deploy_modal)

    # App library commands
    app_list_parser = subparsers.add_parser("app-list", help="List apps in library")
    app_list_parser.set_defaults(func=cmd_app_list)

    app_show_parser = subparsers.add_parser("app-show", help="Show app code")
    app_show_parser.add_argument("name", help="App name")
    app_show_parser.set_defaults(func=cmd_app_show)

    app_deploy_parser = subparsers.add_parser("app-deploy", help="Deploy app from library")
    app_deploy_parser.add_argument("name", help="App name")
    app_deploy_parser.add_argument("--gpu", help="GPU type override")
    app_deploy_parser.add_argument("--secrets", nargs="*", default=[])
    app_deploy_parser.add_argument("--timeout", type=int, default=300)
    app_deploy_parser.set_defaults(func=cmd_app_deploy)

    app_add_parser = subparsers.add_parser("app-add", help="Add app to library")
    app_add_parser.add_argument("name", help="App name")
    app_add_parser.add_argument("--file", required=True, help="App file path")
    app_add_parser.add_argument("--description", help="App description")
    app_add_parser.add_argument("--tags", nargs="*", default=[])
    app_add_parser.add_argument("--gpu", help="Default GPU type")
    app_add_parser.set_defaults(func=cmd_app_add)

    app_update_parser = subparsers.add_parser("app-update", help="Update app in library")
    app_update_parser.add_argument("name", help="App name")
    app_update_parser.add_argument("--file", required=True, help="App file path")
    app_update_parser.set_defaults(func=cmd_app_update)

    app_delete_parser = subparsers.add_parser("app-delete", help="Delete app from library")
    app_delete_parser.add_argument("name", help="App name")
    app_delete_parser.set_defaults(func=cmd_app_delete)

    # Local serve command (Jupyter/local dev support)
    serve_parser = subparsers.add_parser(
        "serve-local", help="Package notebook and (optionally) run a local HTTP server"
    )
    serve_parser.add_argument(
        "path", help="Path to notebook file or directory containing notebook"
    )
    serve_parser.add_argument(
        "--port", default="8080", help="Port to run the local server (default: 8080)"
    )
    serve_parser.add_argument(
        "--run", action="store_true", help="Actually start the server after packaging"
    )
    serve_parser.set_defaults(func=cmd_serve_local)

    # Endpoint registry commands
    ep_list = subparsers.add_parser("endpoints", help="List stored endpoints")
    ep_list.set_defaults(func=lambda _: list_endpoints())

    def _call_func(args):
        url = resolve_endpoint(args.name_or_url)
        if not url:
            logger.error("Endpoint not found or invalid URL")
            return
        payload = None
        if args.json_inline:
            try:
                payload = json.loads(args.json_inline)
            except Exception as e:
                logger.error(f"Invalid inline JSON: {e}")
                return
        elif args.json_file:
            try:
                payload = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to read JSON file: {e}")
                return
        else:
            try:
                raw = input("Enter JSON payload (or leave empty for {}): ").strip()
                if raw:
                    payload = json.loads(raw)
                else:
                    payload = {}
            except Exception as e:
                logger.error(f"Invalid JSON: {e}")
                return
        try:
            resp = http_call(url, args.method, payload)
            print(f"Status: {resp.status_code}")
            try:
                print(json.dumps(resp.json(), indent=2))
            except Exception:
                print(resp.text)
        except Exception as e:
            logger.error(f"HTTP call failed: {e}")

    ep_call = subparsers.add_parser(
        "call", help="Call an endpoint by name or URL with JSON"
    )
    ep_call.add_argument("name_or_url", help="Stored name or full URL")
    ep_call.add_argument("--method", default="POST", help="HTTP method (GET or POST)")
    ep_call.add_argument(
        "--json", dest="json_inline", help="Inline JSON payload as string"
    )
    ep_call.add_argument(
        "--json-file", dest="json_file", help="Path to JSON file payload"
    )
    ep_call.set_defaults(func=_call_func)
    # Doctor command (runs validation)
    doc_parser = subparsers.add_parser(
        "doctor", help="Check environment (Kaggle CLI, gcloud, credentials)"
    )
    doc_parser.set_defaults(func=lambda _: validate_setup())

    # Batch processing commands
    batch_parser = subparsers.add_parser(
        "batch", help="Process multiple notebooks in batch"
    )
    batch_parser.add_argument(
        "input_file", help="File containing list of notebooks to process (one per line)"
    )
    batch_parser.add_argument(
        "--operation",
        choices=["download", "deploy", "run"],
        default="download",
        help="Operation to perform on each notebook",
    )
    batch_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process notebooks in parallel (default: sequential)",
    )
    batch_parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for batch operations (default: current directory)",
    )
    batch_parser.set_defaults(func=cmd_batch)

    # Notebook preprocessing command
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess notebook files"
    )
    preprocess_parser.add_argument(
        "input_path", help="Path to notebook file or directory"
    )
    preprocess_parser.add_argument(
        "--clean-metadata",
        action="store_true",
        help="Remove unnecessary metadata from notebook",
    )
    preprocess_parser.add_argument(
        "--remove-outputs",
        action="store_true",
        help="Remove output cells from notebook",
    )
    preprocess_parser.add_argument(
        "--scan-security",
        action="store_true",
        help="Scan for hardcoded credentials and security issues",
    )
    preprocess_parser.add_argument(
        "--optimize-imports",
        action="store_true",
        help="Analyze and optimize import statements",
    )
    preprocess_parser.add_argument(
        "--output", help="Output path for processed notebook (default: overwrite input)"
    )
    preprocess_parser.set_defaults(func=cmd_preprocess)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nFor detailed help on a command, use: python runna.py <command> --help")
        return

    try:
        # Ensure we have the function attribute
        if not hasattr(args, "func"):
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            return

        args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if logger.level == logging.DEBUG:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Add a --validate flag for setup checking
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        validate_setup()
    else:
        main()
