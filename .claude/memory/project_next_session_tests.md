---
name: Next session — narrowed test approach for this project
description: Immediate next step is organizing tests for the URC app within this repo (not the full test-agent framework which is a separate project)
type: project
---

The full Splunk test agent (Claude Agent SDK + MCP) is scoped as a **separate project** — see `/workspace/docs/splunk-test-agent-proposal.md`.

**For this repo/project, the narrowed approach is:**
- Organize existing Playwright tests (`tests/e2e/playwright/`) to cover the new GUI polish features (sorting, search, pagination, clone, tag input, import/export, error status)
- Add targeted E2E tests that exercise the new UI without the full AI agent framework
- Use the existing Taskfile `test:lifecycle` patterns
- Keep it simple: Playwright + Splunk REST helpers (already in `tests/e2e/playwright/helpers.ts`)

**What was just completed (gui-polish spec):**
- 22 tasks, 161 tests, 95% coverage on new code
- 47 files changed, +4,908 lines
- All unit tests are in `react/packages/urc-builder/src/{hooks,utils,builder/components}/*.unit.ts`
- No E2E/integration tests yet for the new features

**How to apply:** Start next session by writing Playwright E2E tests that cover the critical new GUI flows against a live Splunk instance.
