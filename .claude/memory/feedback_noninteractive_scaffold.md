---
name: Non-interactive @splunk/create scaffold
description: Use Taskfile vars for non-interactive React scaffolding; @splunk/create is a devDep, not npx-cache-resolved
type: feedback
---

`@splunk/create` is a devDependency of the `react/` monorepo root. The non-interactive wrapper at `react/bin/splunk-create-noninteractive.mjs` imports generators programmatically — no npx cache hacks.

**Why:** @splunk/create CLI only exposes appName + --mode; componentType, pageName, generatorMode require programmatic access. Installing as devDep keeps imports clean and version-locked.

**How to apply:** Taskfile tasks accept variables for non-interactive use, fall back to interactive when omitted:
- `task react:create APP_NAME=MyApp` — scaffolds app + page + component
- `task react:add-page PAGE_NAME=builder APP_NAME=my-app` — adds page to existing app
- `task react:add-component COMPONENT_NAME=MyWidget` — creates component library
- `react:*` tasks work for both standalone apps and UCC overlay packages — the React build just produces `stage/`, and the merge step (`ucc:merge-ui`) handles overlay into UCC output.
