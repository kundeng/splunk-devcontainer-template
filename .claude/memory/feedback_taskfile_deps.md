---
name: Dependencies install via Taskfile into app lib dir
description: Never pip install globally; all Python deps must go through Taskfile tasks into the UCC app's package/lib directory
type: feedback
---

All Python dependencies must be installed into the UCC app's `package/lib/` directory, not globally via `pip install`. Use Taskfile tasks for dependency management.

**Why:** The app runs inside Splunk's embedded Python, not the system Python. Dependencies must be vendored/installed into the app package so they ship with the .tar.gz. Global pip installs are useless for the actual runtime.

**How to apply:** Always use the relevant Taskfile task (e.g., `ucc:build` which handles `requirements.txt` → `package/lib/`). Never run bare `pip install` for app dependencies.
