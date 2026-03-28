---
name: Use Taskfile primitives as a developer
description: Always use task commands (task dev:start, task ucc:build, etc.) instead of raw shell commands to dogfood and improve them
type: feedback
---

Use Taskfile task primitives as the primary interface for all dev workflows — building, testing, staging, deps, etc. — as a real developer would.

**Why:** The Taskfile tasks are the template's developer experience. By using them as the primary interface during URC app development, friction, bugs, and missing capabilities surface naturally. This dogfooding improves the template itself.

**How to apply:** When building/testing the URC app or any work in this repo, prefer `task <name>` over raw docker/python/npm commands. If a task is missing or broken, fix/add it rather than working around it. Report issues with task primitives as part of the development flow.
