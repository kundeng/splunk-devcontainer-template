---
name: Use worktrees for separate goals
description: User prefers git worktrees on different branches to keep template improvements vs app development separate
type: feedback
---

Use git worktrees on separate branches to keep different workstreams isolated:
- Template improvements (UCC alignment, cross-platform) on one branch
- REST API connector app development on another branch

**Why:** Keeps concerns separated and allows parallel progress without merge conflicts between template changes and app-specific code.

**How to apply:** When starting substantial work on either goal, create a dedicated branch and use worktree isolation. Can use Claude Code's `isolation: "worktree"` for agents working on separate goals.
