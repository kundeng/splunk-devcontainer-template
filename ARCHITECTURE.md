# Architecture

A Docker-out-of-Docker development environment for building Splunk applications — custom search commands (Python), React UI apps with Dashboard Studio components, and any combination of backend + frontend Splunk apps.

## What This Template Supports

- **Custom search commands** — Python chunked-protocol commands, modular inputs, REST handlers, alert actions
- **React UI apps** — built with `@splunk/create`, using `@splunk/dashboard-core` and Dashboard Studio exported components
- **Multi-app development** — multiple Splunk apps developed and tested simultaneously
- **Simultaneous dev + staging** — dev on port 8000 (bind-mounted, live reload) and staging on port 18000 (baked-in, self-contained)
- **Python debugging** — VSCode attaches to debugpy inside the Splunk container via SA-VSCode add-on
- **React debugging** — Chrome DevTools via webpack dev server (HMR) or source maps in Splunk Web

## System Overview

```
┌──────────────────────┐     ┌──────────────────────────────────────┐
│  Dev Container       │     │  Docker Compose                      │
│  (tools only)        │     │                                      │
│  Node 22 / Python 3.12    │  splunk-dev     (port 8000, 8089)    │
│  Docker CLI / Go Task│────▶│    bind-mounted apps, skip-provision │
│  AppInspect / ruff   │     │                                      │
│  debugpy             │     │  splunk-staging (port 18000, 18089)  │
└──────────────────────┘     │    baked-in apps, same entrypoint    │
        │                    └──────────────────────────────────────┘
        │                                    ▲
        ├── splunk/config/apps/ ─────────────┘  (bind-mounted + symlinked)
        └── packages/          → builds to → config/apps/*/appserver/static/
```

## Directory Structure

```
.devcontainer/
  devcontainer.json              # Tools-only container (Node, Python, Docker CLI, Task)
  docker-compose.yml             # Dev Splunk (target: dev, bind mounts, port 8000)
  docker-compose.staging.yml     # Staging Splunk (target: staging, baked apps, port 18000)
  post-create.sh                 # Tool install + Splunk image build
splunk/
  Dockerfile                     # Multi-stage: base → dev / staging
  entrypoint-wrapper.sh          # Skip-provision + auto-discover /tmp/apps/*.tgz for SPLUNK_APPS_URL
  config/
    apps/                        # Splunk app source directories (symlinked into Splunk)
      splunk-config-dev/         # Dev-mode web.conf (js_no_cache, enableWebDebug, etc.)
      <your_app>/                  # Created via `task app:create APP_NAME=<your_app>`
        bin/                     # Python scripts (custom commands, modular inputs)
          lib/                   # Shared Python libraries
        default/                 # commands.conf, searchbnf.conf, restmap.conf, app.conf
        metadata/
    deps.yml                     # Splunkbase dependencies (MLTK, SA-VSCode, etc.)
  stage/                         # Built tarballs (.tgz) — gitignored, created on demand
packages/                        # React/JS source (monorepo via @splunk/create) — created on demand
  <frontend-app>/                # Scaffolded with npx @splunk/create
    src/main/
      webapp/pages/              # React pages (JSX + Dashboard Studio definitions)
      resources/splunk/          # Splunk app skeleton (views, nav)
    package.json
    webpack.config.js            # Builds to config/apps/<app>/appserver/static/
Taskfile.yml                     # All automation
.env                             # Secrets/config — gitignored
splunk.env.example               # Template for .env
AGENTS.md                        # Concise repo summary + windloop hint
ARCHITECTURE.md                  # This file — full architecture docs
.vscode/launch.json              # Debug configs (Python attach, Chrome, React dev server)
tests/e2e/                       # E2E test scripts (devcontainer + Splunk lifecycle)
.ref/                            # Reference repos (docker-splunk, kveditor) — gitignored
```

## Environment Setup

Copy `splunk.env.example` to `.env` and fill in your values. `.env` is gitignored.

```bash
cp splunk.env.example .env
```

### Devcontainer (recommended)

`LOCAL_WORKSPACE_FOLDER` is injected automatically via `runArgs` in `devcontainer.json` — no manual setup needed. It resolves to the Mac host path of this repo, which is required so `docker compose` bind mounts point to the correct location on the Docker host.

### Native Mac (without devcontainer)

`docker compose` resolves relative paths in `docker-compose.yml` relative to the compose file location on the Mac host — so bind mounts work correctly without any extra configuration. Just run `task` commands directly from the repo root.

---

### Example: App created via `task app:create`

```
splunk/config/apps/
  splunk-config-dev/                     # Dev settings (always present, ships with template)
  my_custom_app/                         # Created via: task app:create APP_NAME=my_custom_app
    bin/                                 # Python scripts (custom commands, modular inputs)
    default/
      app.conf                           # [package] id = my_custom_app, [ui] label = my custom app
      commands.conf                      # Custom search command definitions
      data/ui/views/                     # Dashboard XML views
      data/ui/nav/default.xml
    metadata/default.meta
```

## App Naming Model

`APP_NAME` is the single identifier that ties everything together:

| Concept | Value | Location |
|---|---|---|
| **Folder name** | `APP_NAME` | `splunk/config/apps/<APP_NAME>/` |
| **Package ID** (`app.conf [package] id`) | `APP_NAME` | Splunkbase, REST API |
| **UI Label** (`app.conf [ui] label`) | `APP_NAME` with underscores → spaces | Splunk Web |
| **React source** (if applicable) | `APP_NAME` | `packages/<APP_NAME>/` |

Set `APP_NAME` in `.env` to identify the **primary app** you're developing. All `app:*`, `react:*`, and CI/CD commands use it as the default when no explicit `APP_NAME=` argument is passed.

For React-based apps, the source lives in `packages/<APP_NAME>/` and the built output lands in `splunk/config/apps/<APP_NAME>/appserver/static/`. The `react:build-install` command bridges the two.

Helper apps (like `splunk-config-dev`) are always present and don't need `APP_NAME`.

## Commands

### Dev Splunk Lifecycle

```bash
task splunk:build           # build dev Splunk image (first time / Dockerfile change)
task splunk:up              # start + sync app symlinks (no rebuild)
task splunk:refresh         # reload configs + static assets via REST API (~2s)
task splunk:restartd        # restart splunkd process (~10s)
task splunk:restart         # restart container (skip-provision ~30s)
task splunk:reprovision     # force full Ansible re-provisioning
task splunk:down            # stop container
task splunk:clean           # stop + remove volumes (full reset)
task splunk:logs            # follow container logs
task splunk:status          # check container status
```

### Staging Splunk (port 18000) — runs alongside dev

```bash
task splunk:build-staging   # build staging image (apps baked in from splunk/stage/)
task splunk:up-staging      # start staging (port 18000)
task splunk:down-staging    # stop staging container
task splunk:clean-staging   # stop staging + remove volumes
task splunk:logs-staging    # follow staging container logs
```

### App Development

```bash
task app:create APP_NAME=x  # scaffold + symlink + refresh Splunk (no recreate)
task app:sync-links         # create/update symlinks in Splunk container
task app:provision          # package + install app(s) via splunk CLI
task app:package APP_NAME=x # package to splunk/stage/x.tgz (for staging only)
```

**Which one do I need?**

| Situation | Command |
|---|---|
| New app just created, want it visible in Splunk | `app:create` (calls sync-links automatically) |
| Symlinks got out of sync (e.g. after `splunk:clean`) | `app:sync-links` |
| App needs full install (Python deps, restmap, etc.) | `app:provision` |
| Preparing a release build for staging | `app:package` → `splunk:build-staging` |

- **`sync-links`** — fastest (~1s). Makes the app *visible* via symlink. Enough for `.conf` edits, dashboards, and Python scripts (no packaging needed). Called automatically by `splunk:up` and `app:create`.
- **`provision`** — packages the app into a `.tgz` then runs `splunk install app` inside the container. Use this when Splunk needs to fully register the app (e.g. first install of a custom search command, REST handler, or alert action). Does **not** build React — use `react:build-install` for that.
- **`package`** — only produces a `.tgz` in `splunk/stage/`. Used as input for staging image builds (`splunk:build-staging`). Not needed for day-to-day dev.

### React UI

```bash
task react:create           # scaffold with npx @splunk/create + symlink
task react:start            # dev server with HMR (port 3000)
task react:build-install    # build + copy to appserver/static + package + install
```

### Dependencies, Python & Testing

```bash
task deps:install           # install Splunkbase deps from deps.yml (idempotent)
task python:lint            # ruff check
task python:format          # ruff format
task python:test            # pytest
task test:e2e               # full E2E test (devcontainer + Splunk lifecycle)
task test:all               # lint + E2E
```

## Developer Workflow

### First-Time Setup

1. Open repo in VS Code → "Reopen in Container"
2. `post-create.sh` installs tools and builds the Splunk image
3. `task splunk:up` → starts Splunk, syncs app symlinks (full Ansible provisioning on first boot, ~50s)
4. `task deps:install` → install Splunkbase dependencies (SA-VSCode, MLTK, etc.)

### Creating a New App

1. `task app:create APP_NAME=my_app` → scaffolds app, creates symlink, refreshes Splunk (~2-10s, no container recreation)
2. App is immediately visible in Splunk
3. Edit `.conf` files → `task splunk:refresh` to reload (~2s, no restart)

### Creating a React App

1. `APP_NAME=my_app task react:create` → scaffolds React app + Splunk app skeleton, syncs symlinks, refreshes Splunk (~2-10s, no container recreation)
2. `task react:start` → webpack dev server with HMR on :3000
3. `task react:build-install` → build, copy to `appserver/static/`, install into Splunk

### Daily Dev Loop (fastest → slowest)

| Change | Command | Time |
|--------|---------|------|
| Edit `.conf` / dashboard | `task splunk:refresh` | ~2s |
| Edit Python code | `task splunk:restartd` | ~10s |
| New app | `task app:create` (symlink + refresh) | ~2-10s |
| Dockerfile change | `task splunk:build` then `task splunk:up` | varies |
| Full reset | `task splunk:clean` then `task splunk:up` | ~90s |

### Staging Verification

1. `task app:package APP_NAME=my_app` → package apps into `splunk/stage/`
2. `task splunk:build-staging` → build staging image with apps baked in
3. `task splunk:up-staging` → start staging on ports 18000/18089/18088

## How It Works

### Multi-Stage Dockerfile

The `splunk/Dockerfile` has three stages:

- **`base`** — installs acl + entrypoint wrapper on the official Splunk image
- **`dev`** (default for `docker-compose.yml`) — entrypoint-wrapper for skip-provision; apps bind-mounted live
- **`staging`** (default for `docker-compose.staging.yml`) — bakes all `.tgz` from `splunk/stage/`; same `entrypoint-wrapper.sh` auto-discovers and sets `SPLUNK_APPS_URL`; self-contained

### Dev + Staging Side by Side

Dev and staging run as separate compose projects with different ports:

| | Dev | Staging |
|---|---|---|
| Compose file | `docker-compose.yml` | `docker-compose.staging.yml` |
| Container | `splunk-dev` | `splunk-staging` |
| Web | `:8000` | `:18000` |
| REST API | `:8089` | `:18089` |
| HEC | `:8088` | `:18088` |
| Apps | Bind-mounted (live edit) | Baked into image |
| Entrypoint | entrypoint-wrapper.sh | entrypoint-wrapper.sh (same) |

### Skip-Provision Entrypoint

1. **First start**: No marker → full Ansible → writes `/opt/splunk/var/.provisioned`
2. **Subsequent starts**: Marker exists → starts splunkd directly (~30s vs ~90s)
3. **Force reprovision**: `task splunk:reprovision` removes marker and restarts
4. **Clean reset**: `task splunk:clean` removes volumes (including marker)

The wrapper also writes the Docker healthcheck state file so `checkstate.sh` passes in skip-provision mode.

### Build / Up Separation

The Splunk image is built once during `post-create.sh` (or explicitly via `task splunk:build`). `task splunk:up` only starts the container — it does not rebuild. This avoids unnecessary rebuilds during daily development.

### App Symlink Mounts

The entire `splunk/config/apps/` directory is bind-mounted to `/opt/splunk/dev-apps/` inside the container. Symlinks in `/opt/splunk/etc/apps/` point to each app under `/opt/splunk/dev-apps/`. This means:

- **No container recreation** when adding a new app — just `docker exec ln -s` + refresh
- **Live editing** — file changes on the host are immediately visible through the symlink
- `task app:sync-links` manages symlinks (creates missing, removes stale)
- This pattern is validated by Splunk's own `@splunk/create` tooling (`yarn run link:app`)

| Change type | Action needed |
|---|---|
| `.conf` files, dashboards | `task splunk:refresh` (~2s, no restart) |
| Python code in `bin/` | Re-run the search command (no restart) |
| Python code needing restart | `task splunk:restartd` (~10s) |
| React source | Webpack HMR via `task react:start` (instant) |
| React build for Splunk Web | `task react:build-install` |
| Dashboard Studio JSON | Edit `definition.json` → HMR picks it up |
| New app added | `task app:create` (symlink + refresh ~2-10s) |

### Persistence

- **`splunk-var` volume** — indexed data, KV store, search artifacts, provisioning marker
- **`splunk-etc` volume** — Splunk config, installed apps, and app symlinks; persists across container recreation
- **`/opt/splunk/dev-apps/`** — bind mount of `splunk/config/apps/` (live source)
- **`splunk/stage/`** — built tarballs mounted to `/tmp/apps` in the container

`task splunk:clean` removes all volumes for a full reset.

### React + Dashboard Studio Dev Loop

1. Scaffold: `npx @splunk/create` → creates monorepo under `packages/`
2. Install dashboard packages: `yarn add @splunk/dashboard-core @splunk/dashboard-presets @splunk/dashboard-context`
3. Export dashboard definition from Dashboard Studio → save as `definition.json`
4. Import in JSX: `<DashboardContextProvider preset={...} initialDefinition={definition}><DashboardCore /></DashboardContextProvider>`
5. Dev: `task react:start` → HMR on `:3000`, proxying API calls to Splunk `:8089`
6. Build: `task react:build-install` → webpack output → `appserver/static/` → package → install

### Dependency Management

`splunk/config/deps.yml` declares Splunkbase dependencies. `task deps:install` downloads and installs them idempotently. Default deps include:

- **Python for Scientific Computing** — NumPy/SciPy for custom commands
- **Splunk ML Toolkit** — ML framework
- **SA-VSCode** — enables Python debugging from VSCode

## Debugging

### Python (custom search commands, modular inputs, REST handlers)

Uses the **SA-VSCode** add-on + **debugpy** for remote debugging:

1. Install SA-VSCode: `task deps:install` (included in `deps.yml`)
2. Add debug hook to your Python code:
   ```python
   import sys, os
   sys.path.append(os.path.join(os.environ['SPLUNK_HOME'], 'etc', 'apps', 'SA-VSCode', 'bin'))
   import splunk_debug as dbg
   dbg.enable_debugging(timeout=25)
   ```
3. Trigger the code (run a search, enable an input, etc.)
4. In VSCode: Run → **"Python: Attach to Splunk (debugpy)"**
5. Debugger connects to `localhost:5678` (forwarded from Splunk container)

Path mapping: `splunk/config/apps/<app>/bin/` ↔ `/opt/splunk/etc/apps/<app>/bin/`

### React / JavaScript

**Dev server mode** (recommended for development):
1. `task react:start` → webpack dev server on `:3000` with HMR
2. VSCode: Run → **"Chrome: React Dev Server"**
3. Full source map support, hot reload, breakpoints in JSX

**Splunk Web mode** (for testing the built app):
1. `task react:build-install` → builds and installs into Splunk
2. VSCode: Run → **"Chrome: Splunk Web App"**
3. Source maps from webpack provide debugging in Splunk Web

**Staging mode**:
1. `task splunk:up-staging` → staging on `:18000`
2. VSCode: Run → **"Chrome: Staging Splunk Web"**

### splunk-config-dev

The `splunk-config-dev` app sets `web.conf` options essential for debugging:
- `js_no_cache = True` — disables JS caching
- `minify_js = False` — preserves readable JS
- `enableWebDebug = True` — enables Splunk Web debug mode
- `minify_css = False` — preserves readable CSS

This app is always bind-mounted in dev. Do not include it in staging builds.

## Environment Variables

**Required** (in `.env`):
- `SPLUNK_PASSWORD` — Splunk admin password

**Optional**:
- `APP_NAME` — default app for tasks
- `SPLUNKBASE_USERNAME` / `SPLUNKBASE_PASSWORD` — for Splunkbase downloads
- `SPLUNK_APPS_URL` — comma-separated URLs for first-run app install
- `SPLUNK_VERSION` — Splunk image version (default: 9.4.0)

## Ports

| Port | Service |
|------|---------|
| 8000 | Splunk Web (dev) |
| 8089 | splunkd REST API (dev) |
| 8088 | HTTP Event Collector (dev) |
| 18000 | Splunk Web (staging) |
| 18089 | splunkd REST API (staging) |
| 18088 | HTTP Event Collector (staging) |
| 5678 | Python debugpy (SA-VSCode) |
| 3000 | React dev server (webpack HMR) |

## Troubleshooting

- **Splunk won't start**: `task splunk:logs` → check for errors
- **App changes not visible**: `task splunk:refresh` (reloads configs without restart); if that doesn't work, `task splunk:restartd`
- **New app not visible**: `task app:sync-links` creates missing symlinks; `task splunk:refresh` reloads configs
- **Python debugger won't connect**: ensure SA-VSCode is installed (`task deps:install`), code has `dbg.enable_debugging()`, and port 5678 is exposed
- **React HMR not working**: check webpack proxy config points to `splunk-dev:8089`
- **Corrupt state after restart**: `task splunk:reprovision` (forces full Ansible)
- **Full reset**: `task splunk:clean && task splunk:up`
- **Port conflicts**: edit ports in `.devcontainer/docker-compose.yml` or `docker-compose.staging.yml`
