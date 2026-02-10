# Splunk Dev Container Template

A Docker-out-of-Docker development environment for Splunk applications. The devcontainer provides IDE tooling; Splunk and optional sidecars run as separate Docker Compose services.

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
        └── splunk/config/apps/ ───┘  (bind-mounted live)
```

**Key innovation**: A custom entrypoint wrapper (`splunk/entrypoint-wrapper.sh`) detects whether Splunk has been provisioned (marker file in `splunk-var` volume). On restart, it skips Ansible and starts splunkd directly (~15s vs 90s).

## Directory Structure

```
.devcontainer/
  devcontainer.json            # Tools-only dev container (Node, Python, Docker CLI, Task)
  docker-compose.yml           # Splunk + optional sidecars
  docker-compose.apps.yml      # Auto-generated app bind mounts (gitignored)
  post-create.sh               # Minimal tool installation
splunk/
  Dockerfile                   # Custom Splunk image (stock + entrypoint wrapper)
  entrypoint-wrapper.sh        # Skip-provision logic for fast restarts
  config/
    apps/                      # Splunk app source directories (multi-app)
      splunk-config-dev/       # Dev-mode Splunk settings
    deps.yml                   # Declarative Splunkbase dependency list
  stage/                       # Built tarballs (.tgz) — gitignored
packages/                      # React/JS source (monorepo-friendly)
Taskfile.yml                   # All automation (splunk:*, app:*, react:*, deps:*, python:*)
.env                           # Secrets/config — gitignored
splunk.env.example             # Template for .env
```

## Commands

### Splunk Lifecycle

```bash
task splunk:up              # generate mounts + build image + start
task splunk:down            # stop containers
task splunk:clean           # stop + remove volumes (full reset)
task splunk:restartd        # restart splunkd inside container (~10s)
task splunk:restart         # restart container (wrapper skips Ansible)
task splunk:reprovision     # force full Ansible re-provisioning
task splunk:logs            # follow container logs
task splunk:status          # check container status
```

### App Development (multi-app)

```bash
task app:create APP_NAME=x  # scaffold new app at splunk/config/apps/x
task app:package APP_NAME=x # package to splunk/stage/x.tgz
task app:provision          # package + install all local apps
task app:generate-mounts    # regenerate docker-compose.apps.yml
```

### Dependencies (Splunkbase)

```bash
task deps:install           # install deps from splunk/config/deps.yml (idempotent)
```

### React UI

```bash
task react:create           # scaffold with npx @splunk/create
task react:start            # dev server with HMR (port 3000)
task react:build-install    # build + copy to app + package + install
```

### Python

```bash
task python:lint            # ruff check
task python:format          # ruff format
task python:test            # pytest
```

## How It Works

### Skip-Provision Entrypoint

The custom Splunk image wraps the stock entrypoint:
1. **First start**: No marker → delegates to stock entrypoint (full Ansible) → writes `/opt/splunk/var/.provisioned`
2. **Subsequent starts**: Marker exists → starts splunkd directly, skipping Ansible
3. **Force reprovision**: `task splunk:reprovision` removes marker and restarts
4. **Clean reset**: `task splunk:clean` removes volumes (including marker)

### App Bind Mounts

`task splunk:up` auto-generates `docker-compose.apps.yml` with bind mounts for every directory in `splunk/config/apps/`. Changes to `.conf` files require `task splunk:restartd`; static asset changes are picked up on browser refresh.

### Dependency Management

`splunk/config/deps.yml` declares Splunkbase app dependencies. `task deps:install` downloads and installs them idempotently — skipping apps already at the correct version.

### React Dev Loop

- **Development**: `task react:start` → webpack dev server on :3000 with HMR, proxying to Splunk REST API on :8089
- **Production**: `task react:build-install` → build → copy to `appserver/static/` → package → install

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
| 8000 | Splunk Web |
| 8089 | splunkd REST API |
| 8088 | HTTP Event Collector |
| 3000 | React dev server |

## Troubleshooting

- **Splunk won't start**: `task splunk:logs` → check for errors
- **App changes not visible**: `task splunk:restartd` (conf changes need restart)
- **Corrupt state after restart**: `task splunk:reprovision` (forces full Ansible)
- **Full reset**: `task splunk:clean && task splunk:up`
- **Port conflicts**: Edit ports in `.devcontainer/docker-compose.yml`

## Windloop

This project uses the `spec-driven-dev` skill for autonomous development. Run `/spec-help` to get started.
