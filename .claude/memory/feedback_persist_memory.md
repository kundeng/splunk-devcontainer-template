---
name: Persist memory in project directory
description: User wants Claude memory stored inside /workspace/.claude/memory/ so it survives container rebuilds
type: feedback
---

Store all Claude memory files inside `/workspace/.claude/memory/` (within the project repo), not in `~/.claude/projects/`.

**Why:** The dev container home directory is ephemeral — wiped on rebuild. The project directory is bind-mounted from the host and persists.

**How to apply:** Always write memory files to `/workspace/.claude/memory/` and maintain the symlink from `~/.claude/projects/-workspace/memory` → `/workspace/.claude/memory/`.
