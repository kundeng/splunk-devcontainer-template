# Splunk Dev Container Template

A Docker-out-of-Docker development environment for building Splunk applications. The devcontainer provides IDE tooling (Node 22, Python 3.12, Docker CLI, Go Task); Splunk Enterprise and optional sidecars run as separate Docker Compose services.

## Quick Start

1. **Clone and open in VS Code**:
   ```bash
   git clone <repository-url>
   code splunk-devcontainer-template
   ```

2. **Reopen in Container** — click the prompt or use Command Palette → *Dev Containers: Reopen in Container*

3. **Configure** — edit `.env` (created from `splunk.env.example` on first run):
   ```bash
   SPLUNK_PASSWORD=admin123   # required
   APP_NAME=my_app            # optional default for tasks
   ```

4. **Start Splunk**:
   ```bash
   task dev:up        # starts Splunk + syncs app symlinks (image already built by post-create)
   ```

5. **Access Splunk Web**: http://localhost:8000 (admin / your password)

## Architecture

```
┌──────────────────┐     ┌──────────────────────────┐
│  Dev Container   │     │  Docker Compose Stack     │
│  Node 22         │────▶│  Splunk Enterprise 9.4    │
│  Python 3.12     │     │  (custom image: skip-     │
│  Docker CLI      │     │   provision on restart)   │
│  Go Task         │     │  UF / Cribl / DB (opt)    │
│  AppInspect      │     └──────────────────────────┘
└──────────────────┘               ▲
        │                          │
        └── splunk/config/apps/ ───┘  (bind-mounted + symlinked)
```

A custom entrypoint wrapper detects whether Splunk has already been provisioned. On restart it skips Ansible and starts splunkd directly (~30s vs ~90s).

## Project Structure

```
.devcontainer/
  devcontainer.json            # Tools-only dev container
  docker-compose.yml           # Dev Splunk (target: dev, bind mounts, port 8000)
  docker-compose.staging.yml   # Staging Splunk (reuses dev image, mounts splunk/stage/, port 18000)
  post-create.sh               # Tool install + Splunk image build
splunk/
  Dockerfile                   # Multi-stage: base → dev
  entrypoint-wrapper.sh        # Skip-provision + auto-discover /tmp/apps/*.tgz for SPLUNK_APPS_URL
  config/
    apps/                      # Splunk app source directories (symlinked into Splunk)
      splunk-config-dev/       # Dev-mode Splunk settings (js_no_cache, enableWebDebug)
    deps.yml                   # Declarative Splunkbase dependency list
  stage/                       # Built tarballs (.tgz) — gitignored
react/                         # React/JS source (monorepo via @splunk/create)
Taskfile.yml                   # All automation
tests/e2e/                     # E2E test scripts (devcontainer + lifecycle)
.env                           # Secrets/config — gitignored
splunk.env.example             # Template for .env
```

## Developer Workflow

### First-Time Setup

1. Open repo in VS Code → "Reopen in Container"
2. `post-create.sh` installs tools and builds the Splunk image
3. `task dev:up` → starts Splunk, syncs app symlinks (full Ansible provisioning on first boot, ~50s)
4. `task deps:install` → install Splunkbase dependencies declared in `splunk/config/deps.yml` (optional)

### Creating a New App

1. `task app:create APP_NAME=my_app` → scaffolds app, creates symlink, refreshes Splunk (~2-10s, no container recreation)
2. App is immediately visible in Splunk
3. Edit `.conf` files → `task dev:refresh` to reload (~2s, no restart)

### Creating a React App

1. `task react:create` → scaffolds React app via `@splunk/create`, detects app name, updates `.env`, runs initial build
2. `task react:link` → symlinks `stage/` directly into Splunk `etc/apps/` (run once, or after `dev:clean`)
3. `task react:start` → webpack watch; `stage/` updates live through the symlink — no copy needed
4. `task react:add-page` → add more pages/components interactively via `@splunk/create`
5. `task react:package` → build + package `stage/` as tgz for staging

### Daily Dev Loop (fastest → slowest)

| Change | Command | Time |
|--------|---------|------|
| Edit `.conf` / dashboard | `task dev:refresh` | ~2s |
| Edit Python code | `task dev:restartd` | ~10s |
| New app | `task app:create` (symlink + refresh) | ~2-10s |
| Dockerfile change | `task dev:build` then `task dev:up` | varies |
| Full reset | `task dev:clean` then `task dev:up` | ~90s |

### Staging Verification

1. `task app:package APP_NAME=my_app` → package Python apps into `splunk/stage/`
1. `task react:package` → build + package React app into `splunk/stage/`
2. `task stage:up` → start staging on ports 18000/18089/18088 (auto-installs tgz on first start)

### Installing Splunkbase Dependencies

Declare dependencies in `splunk/config/deps.yml`:

```yaml
dependencies:
  - name: Splunk_SA_Scientific_Python_linux_x86_64
    splunkbase_id: 2882
    version: "4.2.3"
  - name: Splunk_ML_Toolkit
    splunkbase_id: 2890
    version: "5.6.1"
```

Then run `task deps:install`. It downloads and installs only what's missing or outdated.

## Commands

### Dev Splunk Lifecycle

```bash
task dev:build              # build dev Splunk image (first time / Dockerfile change)
task dev:up                 # start + sync app symlinks (no rebuild)
task dev:refresh            # reload configs + static assets via REST API (~2s)
task dev:restartd           # restart splunkd process (~10s)
task dev:restart            # restart container (skip-provision ~30s)
task dev:reprovision        # force full Ansible re-provisioning
task dev:down               # stop container
task dev:clean              # stop + remove volumes (full reset)
task dev:logs               # follow container logs
task dev:status             # check container status
```

### Staging Splunk

```bash
task stage:build            # build staging image (reuses dev image)
task stage:up               # start staging (port 18000)
task stage:down             # stop staging container
task stage:clean            # stop staging + remove volumes
task stage:logs             # follow staging container logs
```

### App Development

```bash
task app:create APP_NAME=x  # scaffold + symlink + refresh Splunk (no recreate)
task app:package APP_NAME=x # package to splunk/stage/x.tgz
task app:sync-links         # create/update symlinks in Splunk container
```

### React UI

```bash
task react:create           # scaffold + initial build, syncs APP_NAME in .env (interactive)
task react:add-page         # add a page/component via @splunk/create (interactive)
task react:link             # symlink stage/ into dev Splunk (auto-builds if stage/ missing)
task react:start            # webpack watch — stage/ updates live via symlink
task react:build            # production webpack build (always rebuilds stage/)
task react:package          # build + package stage/ as splunk/stage/<APP_NAME>.tgz for staging
```

### Python & Testing

```bash
task python:lint            # ruff check
task python:format          # ruff format
task python:test            # pytest
task test:e2e               # full E2E test (devcontainer + Splunk lifecycle)
task test:all               # lint + E2E
```

## How It Works

### Skip-Provision Entrypoint

The custom Splunk image wraps the stock entrypoint:

1. **First start** — no marker → full Ansible provisioning → writes `/opt/splunk/var/.provisioned`
2. **Subsequent starts** — marker exists → starts splunkd directly (~30s vs ~90s)
3. **Force reprovision** — `task dev:reprovision` removes marker and restarts
4. **Clean reset** — `task dev:clean` removes volumes (including marker)

The wrapper also writes the Docker healthcheck state file so `checkstate.sh` passes in skip-provision mode.

### App Symlink Mounts

The entire `splunk/config/apps/` directory is bind-mounted to `/opt/splunk/dev-apps/` inside the container. Symlinks in `/opt/splunk/etc/apps/` point to each app under `/opt/splunk/dev-apps/`. This means:

- **No container recreation** when adding a new app — just `docker exec ln -s` + refresh
- **Live editing** — file changes on the host are immediately visible through the symlink
- `task app:sync-links` manages symlinks (creates missing, removes stale)
- This pattern is validated by Splunk's own `@splunk/create` tooling (`yarn run link:app`)

### Persistence

- **`splunk-var` volume** — indexed data, KV store, search artifacts, provisioning marker
- **`splunk-etc` volume** — Splunk config, installed apps, and app symlinks; persists across container recreation
- **`/opt/splunk/dev-apps/`** — bind mount of `splunk/config/apps/` (live source)
- **`splunk/stage/`** — built tarballs mounted to `/tmp/apps` in the container

`task dev:clean` removes all volumes for a full reset.

### Build / Up Separation

The Splunk image is built once during `post-create.sh` (or explicitly via `task dev:build`). `task dev:up` only starts the container — it does not rebuild. This avoids unnecessary rebuilds during daily development.

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

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SPLUNK_PASSWORD` | Yes | Splunk admin password |
| `APP_NAME` | No | Default app for task commands |
| `SPLUNKBASE_USERNAME` | No | For `task deps:install` with Splunkbase IDs |
| `SPLUNKBASE_PASSWORD` | No | For `task deps:install` with Splunkbase IDs |
| `SPLUNK_APPS_URL` | No | Comma-separated URLs for first-run Ansible install |
| `SPLUNK_VERSION` | No | Splunk image version (default: 9.4.0) |

## CI/CD

`.github/workflows/release.yml` is a starting point for packaging and AppInspect validation. Set repository secrets `SPLUNKBASE_USERNAME` and `SPLUNKBASE_PASSWORD` for Splunkbase publishing.

## Troubleshooting

- **Splunk won't start** — `task dev:logs` → check for errors
- **App changes not visible** — `task dev:refresh` (reloads configs without restart); if that doesn't work, `task dev:restartd`
- **New app not visible** — `task app:sync-links` creates missing symlinks; `task dev:refresh` reloads configs
- **Python debugger won't connect** — ensure SA-VSCode is installed (`task deps:install`), code has `dbg.enable_debugging()`, and port 5678 is exposed
- **Corrupt state after restart** — `task dev:reprovision` (forces full Ansible)
- **Full reset** — `task dev:clean && task dev:up`
- **Port conflicts** — edit ports in `.devcontainer/docker-compose.yml` or `docker-compose.staging.yml`
- **Dev container issues** — rebuild via Command Palette → *Dev Containers: Rebuild Container*

## License

MIT