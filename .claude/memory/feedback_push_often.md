---
name: Push to origin frequently
description: Push commits to remote branches often to prevent corruption from Dropbox sync
type: feedback
---

Push to origin branches frequently for bookkeeping. User runs the workspace from a Dropbox folder and wants to prevent corruption from sync conflicts.

**Why:** Dropbox can corrupt git repos if files change mid-sync. Pushing to remote ensures work is safely backed up.

**How to apply:** After completing a logical unit of work (a few commits), push to origin. Don't batch up many commits before pushing.
