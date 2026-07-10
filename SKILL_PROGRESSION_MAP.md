# Skill Progression Map for gPU

Based on analysis of 329 PRs, review feedback, recurring issues, and codebase patterns.

---

## 1. Narrow Broad Exception Handling → Specific Exception Types

**Evidence:**
- **280+ PRs** (#280–#329) were spawned trying to fix bare `except:` → `except Exception:` in `runna.py`. Only 2 merged (#302, #307); the rest were closed as duplicates — a clear sign of an uncoordinated automation loop.
- Sourcery's review on PR #197 explicitly called out: *"Consider catching more specific exceptions (e.g., `requests.RequestException`) instead of a broad `Exception` so that unexpected failures still surface."* and *"Instead of silently `pass` in `download_remote_notebook` fallback, consider at least logging at debug level."*
- AST analysis of `runna.py` reveals **63 `except Exception` blocks** vs only **24 specific exception handlers**. 17 of those broad handlers silently `pass` with no logging.

**Actionable recommendation:**
- Replace `except Exception: pass` in URL-conversion helpers (lines 701, 717, 729, 741) with `except (ValueError, requests.RequestException): logger.debug(...)`.
- Replace `except Exception: pass` in endpoint registry reads (line 970) with `except (json.JSONDecodeError, OSError)`.
- Replace `except Exception:` in `deploy_to_modal` (line 2097) with `except (subprocess.CalledProcessError, FileNotFoundError)`.
- Add `logger.debug(...)` to every silent fallback — the `download_remote_notebook` fallback at line 863 is a live bug source per `fail.md`.

**Skill to deepen:** Python exception specificity and defensive logging patterns.

---

## 2. Monolith Decomposition — Breaking Up `runna.py`

**Evidence:**
- `runna.py` is **3,112 lines** with **99 functions** and **3 trivial dataclasses**. The `main()` function alone is **374 lines** (a giant if/elif dispatcher).
- 11 functions exceed 50 lines; `deploy_to_aws_lambda` is 125 lines, `cmd_batch` is 92 lines.
- The notebook_ml_orchestrator subpackage was introduced (PRs #4–#7) with proper separation (database, job_queue, backend_router, workflow_engine, batch_processor) but was **reverted** (PR #8) and the code remains in the monolith.
- `fail.md` documents multiple runtime failures that trace back to the tangled pull→push→deploy pipeline being hard to debug.

**Actionable recommendation:**
- Extract `runna.py` into a package: `runna/cli.py` (main dispatcher), `runna/kernels.py` (pull/push/run/create), `runna/deploy.py` (GCF, AWS, Modal), `runna/endpoints.py` (registry, call, send, chat), `runna/convert.py` (nbconvert, dependency detection).
- Keep the single `python3 runna.py` entry point working via `runna/__main__.py`.
- Each module under 300 lines, each function under 40 lines.

**Skill to deepen:** Incremental refactoring of large Python monoliths into packages without breaking CLI interfaces.

---

## 3. Test Infrastructure — Beyond Shell Scripts

**Evidence:**
- Only test files are `test_app_library.sh`, `test_modal.sh`, `test_send.sh` — **shell scripts that call the CLI and check exit codes**. No unit tests, no pytest, no assertions on output content.
- The notebook_ml_orchestrator subpackage (now reverted) included pytest + Hypothesis property-based tests, but they were lost in the revert.
- PR #6 shows @quazfenton applying coderabbitai review comments to test files, indicating interest in improving tests.
- `fail.md` documents a litany of runtime failures (push failures, slug mismatches, download errors) that could have been caught by integration tests.

**Actionable recommendation:**
- Create `tests/` with pytest: `test_endpoint_registry.py` (JSON round-trip, missing file handling), `test_url_conversion.py` (GitHub/GitLab/Bitbucket raw URL conversion), `test_notebook_conversion.py` (magic command stripping, nbconvert fallback), `test_dependency_detection.py` (import scanning).
- Add a `test_cli_smoke.py` using `subprocess.run` to exercise `runna.py doctor`, `runna.py list`, `runna.py app-list` and assert on exit codes and output patterns.
- Add CI via GitHub Actions running `pytest` on every push.

**Skill to deepen:** Building pytest test suites for CLI tools with external service dependencies (mocking kaggle/gcloud/modal).

---

## 4. `.gitignore` and Repository Hygiene

**Evidence:**
- The current `.gitignore` is only 4 lines: `.env*`, `env`, `venv`, `.kaggle` — **missing `__pycache__/`, `*.pyc`, `.kaggle_state/`**.
- `__pycache__/runna.cpython-312.pyc` is committed in the repo.
- Three separate PRs (#55, #56, #58) were opened to add `__pycache__/` and `*.pyc` to `.gitignore`. None merged.
- PR #329 (AGENTS.md) flagged this as a TODO item: *"Add `.kaggle_state/` and `__pycache__/` to `.gitignore`"*.
- Kilo-code-bot review on PR #329 called out: *"Repository name typo: 'gPU' instead of 'GPU'"* and a count mismatch in the auth section.

**Actionable recommendation:**
- Update `.gitignore` to include `__pycache__/`, `*.pyc`, `*.pyo`, `.kaggle_state/`, `*.egg-info/`, `dist/`, `build/`.
- Remove committed `__pycache__/runna.cpython-312.pyc` from git tracking (`git rm --cached`).
- Fix the "gPU" vs "GPU" naming inconsistency flagged in PR #329 review.

**Skill to deepen:** Repository hygiene and preventing drift between documented standards and actual state.

---

## 5. Error Reporting and User Diagnostics

**Evidence:**
- `fail.md` is a 146-line user error log showing repeated failures: push failures with *"Your kernel title does not resolve to the specified id"*, download errors, wrong paths.
- The `doctor.py` command only checks 4 things (Python version, kaggle package, credentials, API connection) but doesn't validate `gcloud`, `aws`, or `modal` setup.
- Several `except Exception` blocks at lines 863, 880, 889, 970, 1173, 1278 swallow errors silently — making it impossible for users to diagnose why a download or deploy failed.
- The auth fallback order issue in `fail.md`: *"change order of auth attempt fallback, kagglehub sdk last because dotenv doesn't recognize .env in project root"*.

**Actionable recommendation:**
- Enhance `doctor.py` to also check: `gcloud` CLI + auth status, `aws` CLI + configured credentials, `modal` CLI + token status, `.env` file loading, `nbconvert` availability.
- Replace all `except Exception: pass` with `except Exception as e: logger.debug(f"...: {e}")` at minimum.
- Add structured error output: when `kernel_push` fails, report the exact Kaggle API error response, not just "Push failed".
- Fix auth order: check env vars → `.env` file → `~/.kaggle/kaggle.json` → kagglehub SDK (per user report in `fail.md`).

**Skill to deepen:** Designing diagnostic and error-reporting patterns for multi-cloud CLI tools.

---

## 6. Automation Loop Discipline — Preventing PR Spam

**Evidence:**
- ~280 PRs were opened for the same "Fix bare except" change (PRs #280–#329). Most were closed as duplicates. Only 2 merged.
- PRs #197 and #66 are still **open** with the same change, creating confusion.
- This pattern suggests the Tembo automation agent was re-triggered on the same issue without checking if a PR already existed.

**Actionable recommendation:**
- Before opening a PR, search `gh pr list` for existing PRs addressing the same files/issue.
- Add a pre-flight check to the automation workflow: if ≥1 open PR already targets the same file, add a comment to the existing PR rather than opening a new one.
- Close stale duplicate PRs #197 and #66 (they duplicate the already-merged #12 and #302/#307).

**Skill to deepen:** Idempotent automation — ensuring agents check existing state before creating new resources.

---

## Summary: Skill Priority Matrix

| # | Skill | Priority | Evidence Strength |
|---|-------|----------|-------------------|
| 1 | Specific exception handling + logging | 🔴 High | 280+ duplicate PRs, 63 broad except blocks, reviewer feedback |
| 2 | Monolith → package decomposition | 🔴 High | 3,112-line file, reverted refactor, runtime failures |
| 3 | Pytest test suite | 🟡 Medium | Only shell-script tests, lost pytest after revert, user error log |
| 4 | .gitignore + repo hygiene | 🟡 Medium | 3 unmerged PRs, committed .pyc, review comments |
| 5 | Error reporting & diagnostics | 🟡 Medium | fail.md, silent exceptions, incomplete doctor command |
| 6 | Automation loop discipline | 🟠 High-for-agent | 280 duplicate PRs is a process smell |

**Start with #1** (exception specificity) — it's the root cause of the PR spam, the silent errors in `fail.md`, and the reviewer feedback. Then move to **#2** (decomposition) which enables **#3** (testing) and **#5** (diagnostics). **#4** and **#6** are quick wins that can be done in parallel.
