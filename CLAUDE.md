# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This repository implements a **Docker-out-of-Docker** development environment for Splunk applications. The architecture separates development tools from the Splunk runtime:

- **Dev Container**: VS Code development environment with tools (Node.js, Python, Go Task, linting)
- **Splunk Container**: Separate Splunk runtime managed via docker-compose
- **Volume Mounting**: Live development through volume-mounted workspace directory

### Key Components

**Development Layer**:
- `.devcontainer/`: VS Code Dev Container configuration
- `Taskfile.yml`: Go Task automation for all operations
- `splunk/packages/`: Volume-mounted directory for single-project development

**Runtime Layer**:
- `docker-compose.yml`: Splunk container orchestration (in .devcontainer/)
- Named volumes: `splunk-etc`, `splunk-var` for data persistence
- Environment variables: `.env` file for configuration

**CI/CD Layer**:
- `.github/workflows/release.yml`: AppInspect validation and release automation

## Common Development Commands

### Essential Workflow
```bash
# Start Splunk runtime
task splunk:up

# Traditional app (non-React)
task app:create APP_NAME=my_app      # Scaffold minimal app under splunk/config/apps/
task app:package APP_NAME=my_app     # Create splunk/stage/my_app.tgz
task app:provision                   # Stage+install apps into running Splunk

# React app (official @splunk/create)
task react:create                    # npx @splunk/create (interactive)
task react:start                     # Start dev server (defaults REACT_PATH from APP_NAME)
task react:build-install             # Build, copy assets into app, package, install
```

### Splunk Management
```bash
task splunk:up              # docker-compose up -d
task splunk:down            # docker-compose down  
task splunk:logs            # docker-compose logs -f splunk
task splunk:status          # docker-compose ps splunk
task splunk:restart         # docker-compose restart splunk
```

### App Development
```bash
task app:create             # Create minimal app (no Yeoman)
task app:package            # Package app to splunk/stage/*.tgz
task app:provision          # Stage+install local app(s) into running Splunk
```

### React Development (Official Splunk CLI)
```bash
task react:create          # Scaffold React Splunk app with @splunk/create
task react:start           # Run React dev server (port 3000 by default)
task react:build-install   # Build React app, copy to appserver/static, package+install
```

## File Structure and Architecture

### Core Configuration Files

**`.devcontainer/devcontainer.json`**:
- Dev container definition with features (docker-outside-of-docker, task)
- VS Code extensions and settings
- Environment variable mapping from host `.env`
- Volume mounting configuration

**`Taskfile.yml`**:
- Central automation with Go Task
- Namespaces: `splunk:*`, `app:*`, `react:*`, `python:*`, `env:*`
- Uses docker compose for Splunk management
- Top-level vars drive paths (compose file, env file, service, dirs)

**`.devcontainer/docker-compose.yml`**:
- Splunk container definition using `splunk/splunk:9.2.1`
- Volume mounting: `./splunk/stage:/tmp/apps` (local app tarballs available at runtime)
- Named volumes for persistence: `splunk-etc`, `splunk-var`
- Environment variables via `.env`, including `SPLUNK_APPS_URL` for app installation
- Minimal Ansible toggles in env: `ANSIBLE_DEPRECATION_WARNINGS=false`, `ANSIBLE_SYSTEM_WARNINGS=false`, `ANSIBLE_ACTION_WARNINGS=false`, `ANSIBLE_PIPELINING=true`

### Development Directories

**`splunk/config/apps/`**:
- Your traditional apps (source of truth)

**`splunk/stage/`**:
- Output tarballs used for install at runtime

**`splunk/config/`**:
- Splunk configuration apps (splunk-config-dev, splunk-config)
- Baked into custom Splunk Docker image for enhanced development

### Environment Variables

**Required in `.env`**:
- `SPLUNK_PASSWORD`: Admin password for Splunk instance

**Optional**:
- `SPLUNKBASE_USERNAME`, `SPLUNKBASE_PASSWORD`: For downloading apps from Splunkbase

## Development Patterns

### Package Mount Strategy (Tarballs)
Place packaged apps (`.tgz`) under `splunk/stage/` on host. These are mounted to the Splunk container at `/tmp/apps` and referenced in `SPLUNK_APPS_URL` for installation by splunk-ansible at startup. This enables:
- Mixing local tarballs and Splunkbase URLs in `SPLUNK_APPS_URL`
- Fast iteration by updating tarballs without rebuilding the image
- Optionally baking tarballs into the image (also copied to `/tmp/apps` during build)

### Task-Based Operations
All operations use Go Task for consistency:
- Discoverable via `task` (default help) or `task --list`
- Parameterized tasks using `{{.CLI_ARGS}}`
- Namespace organization for related commands
- Docker-compose integration for Splunk management

### Persistence Model
- **Splunk Data**: Persisted in named Docker volumes
- **App Code**: Lives in host filesystem (splunk/packages/)
- **Dependencies**: Downloaded to deps/, installed to persistent volumes

## CI/CD

`.github/workflows/release.yml` is provided as a starting point for packaging/validation. Adjust to your needs.

## Project Model

This template focuses on a clear split:
- Traditional app source in `splunk/config/apps/<APP_NAME>`
- React app lives under `packages/<APP_NAME>` (created by `react:create`)
- Built assets are copied into `appserver/static` of the Splunk app

## Dev Container Integration

**Features Used**:
- `docker-outside-of-docker`: Access to host Docker daemon
- `task`: Go Task installation via devcontainer features

**Setup Flow**:
1. Container build with development tools
2. `post-create.sh`: Environment setup, tool installation
3. Task commands become available for Splunk operations

## Troubleshooting Common Issues

**Splunk Not Accessible**:
- Check: `task splunk:status`
- Debug: `task splunk:logs`
- Restart: `task splunk:restart`

**Volume Mount Issues**:
- Ensure workspace directory exists
- Check Docker Desktop file sharing permissions
- Verify container has access to project directory

**AppInspect Failures**:
- Review validation output for specific issues
- Check app.conf configuration
- Ensure proper file permissions and structure

**Port Conflicts**:
- Modify ports in `.devcontainer/docker-compose.yml`
- Default ports: 8000 (Web), 8089 (Management), 3000 (Dev server)

## Modern React/JavaScript Development

This template supports modern Splunk app development with React and Splunk UI framework:

### JavaScript Project Structure (Optional)
```
splunk/packages/MyApp/
├── package.json              # NPM/Yarn configuration
├── yarn.lock / package-lock.json  # Dependency lock files
├── webpack.config.js         # Build configuration
├── src/                      # Source code
│   ├── index.js             # Entry point
│   ├── components/          # React components
│   └── utils/               # Utility functions
├── appserver/               # Splunk app server files
│   └── templates/
└── default/                 # Standard Splunk app structure
    ├── app.conf
    └── data/ui/
```

### Development Workflow for React Apps
1. `task react:create` → scaffold via `@splunk/create`
2. `task splunk:up` → run Splunk (port 8000)
3. `task react:start` → dev server (port 3000)
4. `task react:build-install` → build, copy to `appserver/static`, package+install

### VS Code Debug Configuration
The template includes `.vscode/launch.json` with:
- **Debug Splunk App in Chrome**: Debugs app running in Splunk Web (port 8000)
- **Debug React Dev Server**: Debugs development server (port 3000)
- **Input prompts**: Dynamically asks for app name

### Notes
- Put `APP_NAME=your_app` into `.env`. Tasks default `REACT_PATH` to `packages/$APP_NAME` if present.
- Use `python:*` tasks for lint/format/test if you have Python code.

## Important Development Notes

- **Local Tarballs**: Put `.tgz` packages in `splunk/stage/` and list them in `SPLUNK_APPS_URL` as `/tmp/apps/<name>.tgz`
- **Optional Baking**: App tarballs can be baked into the image and referenced by path
- **Persistent Data**: Splunk configuration and data survive container restarts
- **Clean Resets**: Use `task splunk:clean` to stop and remove named volumes (clean slate)
- **Task Discovery**: Run `task` to see available commands and common flows
- **Ansible Warnings**: The Splunk image installs POSIX ACLs to avoid world-readable temp file warnings; env keeps logs quiet
- **React Development**: Use official @splunk/create CLI for best experience
- **Hot Reload**: Splunk's dev server provides instant updates with proper API integration
- **Official Components**: Use @splunk/react-ui for consistent Splunk look and feel
- **No Manual Config**: Splunk CLI handles all webpack/proxy configuration automatically
- **Production Builds**: Always run `task js:build` before packaging
- **Debugging**: Use VS Code debugger (F5) to debug in Chrome