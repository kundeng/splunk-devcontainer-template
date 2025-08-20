# Splunk App Development Template

A modern, containerized development environment for Splunk applications using **Docker-out-of-Docker** architecture. This template provides VS Code Dev Containers for development tools and docker-compose for Splunk runtime separation. It supports installing local app tarballs at runtime (no need to bake apps into the image).

## Architecture

**Docker-out-of-Docker Pattern**:
- **Dev Container**: Development tools (VS Code, Node.js, Python, Go Task, linting)
- **Splunk Container**: Separate Splunk runtime via docker-compose
- **Shared Workspace**: Volume-mounted development directory

```
Docker Host
├── Dev Container (your workspace)
│   ├── VS Code Server + Extensions
│   ├── Node.js, Python, Go Task
│   └── Development tools
├── Splunk Container (docker-compose)
│   ├── Splunk Enterprise 9.2.1
│   ├── Volume-mounted app
│   └── Persistent data/config
└── Other Services (optional)
```

## Quick Start

1. **Clone and open in VS Code**:
   ```bash
   git clone <repository-url>
   cd dsx_splunk_devtemplate
   code .
   ```

2. **Reopen in Container**: Click "Reopen in Container" when prompted, or use Command Palette → "Dev Containers: Reopen in Container"

3. **Show available actions**:
   ```bash
   task
   ```

4. **Start Splunk runtime**:
   ```bash
   task splunk:up
   ```

5. **Access Splunk Web**: http://localhost:8000 (admin/admin123)

## Key Features

- **Hybrid Architecture**: Dev container + docker-compose for optimal separation
- **Go Task Automation**: Standardized commands for all operations
- **Live Development**: Volume-mounted workspace for real-time changes
- **Persistent Data**: Named volumes preserve Splunk data across restarts
- **Single-App Focus**: One app under `splunk/config/apps/<APP_NAME>`; React assets copied into it
- **Task-driven Provisioning**: Stage and install local apps after the container is up
- **CI/CD Ready**: GitHub Actions for automated testing and releases

## Common Development Tasks

### Splunk Management
```bash
task splunk:up          # Start Splunk via docker-compose
task splunk:down        # Stop Splunk and remove containers
task splunk:logs        # Follow Splunk logs
task splunk:restart     # Restart Splunk service
task splunk:status      # Check Splunk container status
```

### App Development
```bash
task app:create         # Create minimal Splunk app under splunk/config/apps/<APP_NAME>
task app:package        # Package app into splunk/stage/<APP_NAME>.tgz
task app:provision      # Stage+install local app(s) into running Splunk
```

### React Development
```bash
task react:create       # Create new React app with @splunk/create CLI
task react:start        # Start React dev server (uses REACT_PATH; defaults from APP_NAME)
task react:build-install # Build React app, copy to appserver/static, package+install
```

### Code Quality  
```bash
task python:lint        # Lint Python code with flake8
task python:format      # Format with black and isort
task python:test        # Run pytest tests
```

## Project Structure

```
dsx_splunk_devtemplate/
├── .devcontainer/           # Dev container configuration
│   ├── devcontainer.json    # Container definition + features
│   ├── docker-compose.yml   # Splunk runtime definition
│   └── post-create.sh       # Setup script
├── .github/workflows/       # CI/CD pipelines
│   └── release.yml          # Release automation (adjust as needed)
├── splunk/                  # Splunk development workspace
│   ├── config/              # Splunk configuration apps (e.g., splunk-config-dev)
│   │   └── apps/<APP_NAME>/ # Your Splunk app
│   └── stage/               # Packaged app tarballs (.tgz) staged for install
├── .env                     # Environment variables (created from template)
├── splunk.env.example       # Environment template
├── Taskfile.yml             # Go Task definitions
└── README.md                # This file
```

## Environment Configuration

Use `.env` at the repo root to drive docker-compose and task defaults:

```bash
# Required
SPLUNK_PASSWORD=admin123

# Optional (for Splunkbase downloads)
SPLUNKBASE_USERNAME=your_username
SPLUNKBASE_PASSWORD=your_password

# Optional default app name (used by tasks)
APP_NAME=your_app

# Optional app sources (comma-separated) if you want Splunk to install at startup
# Local tarballs (mounted to /tmp/apps) and/or Splunkbase URLs
# SPLUNK_APPS_URL=/tmp/apps/your-app.tgz,https://splunkbase.splunk.com/api/2/apps/2890/releases/5.6.1/download?origin=sb
```

## Development Workflow

### For New React Apps (Recommended)
1. **Create React app with official Splunk CLI**:
   ```bash
   task react:create
   # Or manually: npx @splunk/create
   ```
2. **Start development**:
   - `task splunk:up` - Start Splunk runtime (port 8000)
   - `task react:start` - Start React dev server
3. **Development workflow**:
   - Develop with official Splunk React UI components
   - Hot reload and automatic Splunk API integration
   - Build and install into Splunk: `task react:build-install`

### For Traditional Splunk Apps
1. `task app:create` - Create a minimal Splunk app under `splunk/config/apps/<APP_NAME>`
2. Develop under `splunk/config/apps/<APP_NAME>`
3. `task app:package` - Package to `splunk/stage/<APP_NAME>.tgz`
4. `task app:provision` - Stage + install into running Splunk

## App Installation, Persistence, and Data

- **Local App Tarballs**: Place `.tgz` files under `splunk/stage/` (host). These are mounted to `/tmp/apps` inside the container for install.
- **Prefer Provisioning**: Use `task app:provision` after `splunk:up` instead of baking apps into the image.
- **Persistence**: Installed apps and Splunk configuration/data persist in named volumes (`splunk-etc`, `splunk-var`).
- **Your Project**: Lives in `splunk/config/apps/<APP_NAME>/` locally; package to `.tgz` into `splunk/stage/`.

## CI/CD Integration

GitHub Actions workflow (`.github/workflows/release.yml`) is a starting point for packaging/validation. Adjust to your needs.

Set repository secrets for Splunkbase publishing:
- `SPLUNKBASE_USERNAME`
- `SPLUNKBASE_PASSWORD`

## Behind the Scenes: How It All Works

### Docker-out-of-Docker Architecture Deep Dive

**The Problem**: Traditional dev containers package everything (IDE, tools, runtime) into one container, making it heavy and inflexible.

**The Solution**: Separate development tools from runtime services using Docker-out-of-Docker.

```
┌─────────────────────────────────────────────────────────────┐
│ Docker Host (your machine)                                  │
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │ Dev Container       │    │ Splunk Container            │ │
│  │ (VS Code Server)    │    │ (docker-compose)            │ │
│  │                     │    │                             │ │
│  │ • Node.js 18        │    │ • Splunk Enterprise 9.2.1  │ │
│  │ • Python 3.11       │    │ • Port 8000 (Web UI)       │ │
│  │ • Go Task           │◄──►│ • Port 8089 (Management)    │ │
│  │ • Docker CLI        │    │ • Port 3000 (React dev)    │ │
│  │ • Your code         │    │ • Volume-mounted apps       │ │
│  │                     │    │ • Persistent config/data   │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│           │                               │                 │
│           └───────────────┬───────────────┘                 │
│                           │                                 │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │ Shared Workspace: ./workspace ──► /opt/splunk/etc/apps │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Key Benefits**:
- **Isolation**: Development tools stay separate from runtime
- **Performance**: Each container optimized for its purpose
- **Flexibility**: Easy to swap/upgrade Splunk versions
- **Resource Efficiency**: No duplicate services

### DevContainer Startup Sequence

When you click "Reopen in Container", here's what happens:

1. **Container Creation**:
   ```bash
   # VS Code builds dev container from ubuntu base image
   # Installs features: Docker CLI, Node.js, Python, Task
   ```

2. **Workspace Mounting**:
   ```bash
   # Your local ./workspace is mounted to /workspace in container
   # This enables live file editing between host and container
   ```

3. **Post-Create Setup** (`.devcontainer/post-create.sh`):
   ```bash
   # Installs Splunk dev tools: @splunk/create
   # Installs Python tools: black, flake8, isort, pytest
   
   # Create .env file with default credentials
   cp splunk.env.example .env
   ```

4. **Environment Variables**:
   ```bash
   # DevContainer forwards local environment variables:
   SPLUNKBASE_USERNAME → container
   SPLUNKBASE_PASSWORD → container
   SPLUNK_PASSWORD → container and docker-compose
   ```

### Volume Mounting Magic: Live Development Explained

**The Challenge**: How do file changes in the dev container instantly appear in the running Splunk container?

**The Solution**: Strategic volume mounting at the Docker host level.

```
Host Machine:
splunk/config/apps/<APP_NAME>/
├── appserver/static/     # React build output copied here by react:build-install
└── default/              # Traditional Splunk app configs

Docker Host Volume Layer:
./splunk/stage → Dev Container:/workspace/splunk/stage
./splunk/stage → Splunk Container:/tmp/apps

Build Workflow: Source → Build Process → stage/ → Splunk sees changes
```

**Why This Works**:
1. **Built Output**: Only the `stage/` directory (built Splunk app) is mounted to Splunk
2. **Build Process**: Source code gets built and copied to `stage/` directory
3. **Live Reload**: Splunk automatically detects changes in the mounted `stage/` directory
4. **No File Copying**: Direct filesystem sharing of built assets, not source code

### Official Splunk CLI Tools: @splunk/create Deep Dive

**What `task react:create` actually does**: Runs the official `@splunk/create` CLI to scaffold a Splunk React app, placing it at your configured `REACT_PATH` (defaults from `APP_NAME`).

**Behind `@splunk/create`**:
1. **Interactive Setup**:
   - Prompts for app name, type, features
   - Downloads official Splunk React boilerplate
   - Configures webpack for Splunk deployment

2. **Generated Project Structure**:
   ```
   my_splunk_react_app/
   ├── package.json          # Splunk-optimized dependencies
   ├── webpack.config.js     # Official @splunk/webpack-configs
   ├── src/
   │   ├── main/
   │   │   └── webapp/       # React components
   │   └── main/
   │       └── resources/    # Splunk app structure
   ├── default/
   │   ├── app.conf         # Splunk app configuration
   │   └── data/ui/nav/     # Navigation definitions
   └── static/js/           # Built React assets (production)
   ```

3. **Official Tooling Integration**:
   - `@splunk/react-ui`: Official Splunk UI components
   - `@splunk/webpack-configs`: Pre-configured build pipeline
   - `@splunk/babel-preset`: Splunk-specific transpilation
   - `@splunk/eslint-config`: Linting rules for Splunk apps

### Development Server Magic: How Hot Reload Works

**When you run `task react:start`**:

```bash
# In the generated React app directory (handled by the task):
npm start  # or: yarn start
```

**What Actually Happens**:
1. **Webpack Dev Server Starts**: Usually on port 3000
2. **Splunk API Proxy**: Automatically configured to proxy `/services/` requests to Splunk
3. **Hot Module Replacement**: React components update without page refresh
4. **Source Maps**: Full debugging support in browser dev tools

**The Proxy Configuration** (handled by @splunk/webpack-configs):
```javascript
// Automatically configured, no manual setup needed:
proxy: {
  '/services': {
    target: 'https://localhost:8089',  // Splunk management port
    secure: false,
    changeOrigin: true
  }
}
```

### Notes on AppInspect
Use the Splunk AppInspect CLI (local or Docker image) in your CI/CD as needed. This repo does not include a `task app:validate`.

### Persistent Data Strategy

**The Challenge**: How do you preserve Splunk configuration and data across container restarts?

**The Solution**: Named Docker volumes for persistence + host volumes for development.

```yaml
# docker-compose.yml volumes (current):
volumes:
  - ./splunk/stage:/tmp/apps                    # Local app tarballs available to Splunk at runtime
  - splunk-etc:/opt/splunk/etc                  # Config persistence (named volume)
  - splunk-var:/opt/splunk/var                  # Data persistence (named volume)
```

**What Gets Preserved**:
- **splunk-etc**: User accounts, system configs, installed apps
- **splunk-var**: Indexed data, search artifacts, KV store
- **./splunk/stage**: Built project output (mounted to Splunk, generated from source)

**What Gets Reset**: Container filesystem (logs, temp files, OS-level changes)

### CI/CD Pipeline Internals

**GitHub Actions Workflow** (`.github/workflows/release.yml`):

1. **Trigger Events**:
   ```yaml
   on:
     push:
       branches: [main]     # Validation on every main branch push
       tags: ['v*']         # Release pipeline on version tags
     release:
       types: [published]   # Splunkbase publishing
   ```

2. **Build and Validation Job**:
   ```bash
   # Dynamic app discovery:
   APP_NAME=$(grep -oE '^\s*id\s*=\s*.*' splunk/config/apps/*/default/app.conf | head -1 | awk -F'=' '{print $2}' | xargs)
  APP_VERSION=$(grep -oE '^\s*version\s*=\s*[0-9.]+' splunk/config/apps/*/default/app.conf | head -1 | awk -F'=' '{print $2}' | xargs)
   
   # Smart packaging (excludes dev files):
   tar -czf $APP_NAME-$APP_VERSION.tar.gz \
     --exclude=".git" --exclude=".github" \
     --exclude="build" --exclude=".devcontainer" \
     --exclude="tests" --exclude="node_modules" $APP_NAME
   
   # Cloud-ready validation:
   splunk-appinspect inspect $APP_NAME-$APP_VERSION.tar.gz \
     --mode precert \
     --included-tags cloud \
     --excluded-tags deprecated
   ```

3. **Conditional Publishing**:
   - **Artifacts**: Always uploaded for download
   - **Splunkbase**: Only on version tags with configured secrets

### Task Automation System

**Why Go Task over Make/Scripts**:
- **Cross-platform**: Works on Windows, macOS, Linux
- **YAML Configuration**: Easy to read and maintain
- **Variable Support**: Template variables with `{{.WORKSPACE_DIR}}`
- **Task Dependencies**: Automatic ordering and parallelization

**Task Variable Resolution**:
```yaml
vars:
  PACKAGES_DIR: '/workspace/splunk/packages'  # Available as {{.PACKAGES_DIR}} in all tasks

tasks:
  react:create:
    cmds:
      - |
        cd {{.PACKAGES_DIR}}  # Expands to: cd /workspace/splunk/packages
        npx @splunk/create
```

### Environment Variable Flow

**From Host to Containers**:
```
Host .env → docker-compose (`--env-file .env`) → Splunk container env (used by splunk-ansible)
```

**Detailed Flow**:
1. **Host**: You set `SPLUNK_PASSWORD=mypassword` in `.env`
2. **DevContainer**: Reads via `"${localEnv:SPLUNK_PASSWORD:admin123}"`
3. **Docker Compose**: Receives via `SPLUNK_PASSWORD=${SPLUNK_PASSWORD:-admin123}`
4. **Splunk Container**: Uses as `SPLUNK_START_ARGS=--accept-license`

### Tips & Best Practices

- **React Development**: Use official @splunk/create CLI for best experience
- **Hot Reload**: Splunk's dev server provides instant updates with proper API integration
- **Official Components**: Use @splunk/react-ui for consistent Splunk look and feel
- **Production Build**: Use `task react:build-install` to build, copy assets into your Splunk app, and package/install
- **First Setup**: Container build takes several minutes initially
- **Splunk Access**: Always available at http://localhost:8000 when running
- **Local Tarballs**: Put `.tgz` in `splunk/stage/` and reference via `/tmp/apps/foo.tgz` in `SPLUNK_APPS_URL`
- **Clean Reset**: `task splunk:clean` removes containers and named volumes
- **Task Discovery**: Run `task` to see available commands and flows
- **Volume Persistence**: Named volumes preserve data; host mounts enable live editing
- **Debugging**: Use browser dev tools with React dev server for full debugging
- **AppInspect Early**: Run validation frequently to catch issues early

### Troubleshooting

**Splunk not accessible**: Check `task splunk:status` and `task splunk:logs`
**Port conflicts**: Modify ports in `.devcontainer/docker-compose.yml`
**Permission issues**: Ensure Docker has access to project directory
**Ansible warnings about world-readable temp files**: The image installs POSIX ACLs so Ansible avoids world-readable temp files. Minimal env suppressions in `.devcontainer/docker-compose.yml` keep logs quiet: `ANSIBLE_DEPRECATION_WARNINGS=false`, `ANSIBLE_SYSTEM_WARNINGS=false`, `ANSIBLE_ACTION_WARNINGS=false`, `ANSIBLE_PIPELINING=true`.

## Why APP_NAME?

Sometimes you may want to target only one app directory when running packaging/validation tasks. While this repo is single-app by default, the `APP_NAME` environment variable lets you:

- Selectively package or validate one app: `APP_NAME=my_app task app:package`
- Speed up validation cycles when multiple components/subdirs exist
- Align with CI where the app name may be known ahead of time

If `APP_NAME` is not set, tasks use the default from `.env` or infer from the current app directory name.
**Dev container issues**: Rebuild container via Command Palette

## License

MIT