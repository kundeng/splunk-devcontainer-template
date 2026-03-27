---
name: Git workflow - push to dev, no PRs
description: User is repo owner, prefers pushing to dev branch directly rather than creating PRs
type: feedback
---

Push directly to `dev` branch, no pull requests needed. User is the repo owner and will merge dev → main when ready.

**Why:** PRs add unnecessary ceremony for a solo owner. Direct push to dev is faster.

**How to apply:** When committing work, push to `dev` (or a feature branch if isolating a workstream). Don't create PRs unless explicitly asked.
