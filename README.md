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
   task splunk:up     # builds image, generates app mounts, starts stack
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
        └── splunk/config/apps/ ───┘  (bind-mounted live)
```

A custom entrypoint wrapper detects whether Splunk has already been provisioned. On restart it skips Ansible and starts splunkd directly (~15 s vs ~90 s).

## Project Structure

```
.devcontainer/
  devcontainer.json            # Tools-only dev container
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
Taskfile.yml                   # All automation
.env                           # Secrets/config — gitignored
splunk.env.example             # Template for .env
```

## Commands

### Splunk Lifecycle

```bash
task splunk:up              # generate mounts + build image + start
task splunk:down            # stop containers
task splunk:clean           # stop + remove volumes (full reset)
task splunk:restartd        # restart splunkd inside container (~10 s)
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

## Development Workflows

### Traditional Splunk App

1. `task app:create APP_NAME=my_app` — scaffold under `splunk/config/apps/`
2. Edit `.conf` files, add scripts, etc.
3. `task splunk:restartd` — pick up conf changes (~10 s)
4. `task app:package APP_NAME=my_app` — create tarball for distribution

### React Splunk App

1. `task react:create` — scaffold via `@splunk/create` into `packages/`
2. `task splunk:up` — start Splunk (port 8000)
3. `task react:start` — webpack dev server on port 3000 with HMR, proxying to Splunk REST API on :8089
4. `task react:build-install` — build → copy to `appserver/static/` → package → install

### Installing Splunkbase Dependencies

Declare dependencies in `splunk/config/deps.yml`:

```yaml
dependencies:
  - name: python_for_scientific_computing
    splunkbase_id: 2882
    version: "4.2.3"
  - name: Splunk_ML_Toolkit
    splunkbase_id: 2890
    version: "5.6.1"
```

Then run `task deps:install`. It downloads and installs only what's missing or outdated.

## How It Works

### Skip-Provision Entrypoint

The custom Splunk image wraps the stock entrypoint:

1. **First start** — no marker → full Ansible provisioning → writes `/opt/splunk/var/.provisioned`
2. **Subsequent starts** — marker exists → starts splunkd directly, skipping Ansible
3. **Force reprovision** — `task splunk:reprovision` removes marker and restarts
4. **Clean reset** — `task splunk:clean` removes volumes (including marker)

### App Bind Mounts

`task splunk:up` auto-generates `docker-compose.apps.yml` with a bind mount for every directory in `splunk/config/apps/`. Conf changes require `task splunk:restartd`; static asset changes appear on browser refresh.

### Persistence

- **`splunk-var` volume** — indexed data, KV store, search artifacts, provisioning marker
- **Bind-mounted apps** — live source from `splunk/config/apps/`
- **`splunk/stage/`** — built tarballs mounted to `/tmp/apps` in the container

`task splunk:clean` removes volumes for a full reset.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SPLUNK_PASSWORD` | Yes | Splunk admin password |
| `APP_NAME` | No | Default app for task commands |
| `SPLUNKBASE_USERNAME` | No | For `task deps:install` with Splunkbase IDs |
| `SPLUNKBASE_PASSWORD` | No | For `task deps:install` with Splunkbase IDs |
| `SPLUNK_APPS_URL` | No | Comma-separated URLs for first-run Ansible install |
| `SPLUNK_VERSION` | No | Splunk image version (default: 9.4.0) |

## Ports

| Port | Service |
|------|---------|
| 8000 | Splunk Web |
| 8089 | splunkd REST API |
| 8088 | HTTP Event Collector |
| 3000 | React dev server |

## CI/CD

`.github/workflows/release.yml` is a starting point for packaging and AppInspect validation. Set repository secrets `SPLUNKBASE_USERNAME` and `SPLUNKBASE_PASSWORD` for Splunkbase publishing.

## Troubleshooting

- **Splunk won't start** — `task splunk:logs` → check for errors
- **App changes not visible** — `task splunk:restartd` (conf changes need restart)
- **Corrupt state after restart** — `task splunk:reprovision` (forces full Ansible)
- **Full reset** — `task splunk:clean && task splunk:up`
- **Port conflicts** — edit ports in `.devcontainer/docker-compose.yml`
- **Dev container issues** — rebuild via Command Palette → *Dev Containers: Rebuild Container*

## License

MIT