# Comprehensive Splunk App Development Environment Setup Guide

This guide provides a complete walkthrough for setting up a robust, automated, and containerized development environment for Splunk applications. It leverages VS Code Dev Containers, Docker, Go Task for local automation, and GitHub Actions for CI/CD.

## 1. Introduction & Goals

Developing Splunk apps requires a consistent environment and often involves repetitive tasks. This guide aims to establish a setup that:

- **Standardizes Development:** Ensures all developers use the same tools, Splunk version, and configurations.
- **Simplifies Onboarding:** Allows new team members to get started quickly by simply opening the project in VS Code.
- **Isolates Dependencies:** Confines all necessary tools (Splunk, Python SDK, Node.js, etc.) within a Docker container, avoiding conflicts with local machine setups.
- **Automates Common Tasks:** Uses Go Task for simple commands for operations like starting/stopping Splunk, packaging the app, and creating new apps.
- **Integrates with VS Code:** Leverages VS Code's features for debugging, IntelliSense, and task running.
- **Implements CI/CD:** Automates building, validating, packaging, and releasing the app using GitHub Actions.
- **Manages Configuration Securely:** Provides a mechanism for handling sensitive information like Splunkbase credentials.

## 2. Prerequisites

Before you begin, ensure you have the following installed on your local machine:

- **Visual Studio Code (VS Code):** A source code editor.
- **Docker Desktop:** To build and run Docker containers. Ensure it's running.
- **Git:** For version control.
- **VS Code "Dev Containers" Extension:** (ID: `ms-vscode-remote.remote-containers`). Install this from the VS Code Extensions view.

## 3. Project Initialization & Repository Structure

### 3.1 Create Project Directory and Initialize Git

1. Create a new directory for your Splunk app project:

   ```
   mkdir MySplunkProject
   cd MySplunkProject
   ```

2. Initialize a Git repository:

   ```
   git init
   ```

### 3.2 Recommended Repository Layout

Create the following directory structure within `MySplunkProject`. The actual Splunk app code will reside in a subdirectory (e.g., `MyCoolSplunkApp`), the name of which will be managed by an environment variable.

```
MySplunkProject/
├── .devcontainer/                # VS Code Dev Container configuration
│   ├── devcontainer.json         # Main dev container definition
│   ├── post-create.sh          # Script run after container creation
│   └── custom_setup.sh           # Optional user-defined setup script (e.g., for MLTK, PSC)
├── .github/                      # GitHub specific files
│   └── workflows/                # GitHub Actions workflows
│       └── release.yml           # CI/CD workflow for releases
├── .vscode/                      # VS Code workspace-specific settings (optional)
│   └── settings.json
├── <APP_NAME>/                   # Your Splunk app's root directory (e.g., MyCoolSplunkApp)
│   ├── appserver/
│   ├── bin/
│   ├── default/
│   ├── local/
│   ├── lookups/
│   ├── static/
│   ├── README/
│   └── metadata/
├── docs/                         # Project documentation (optional)
├── scripts/                      # Utility scripts (optional)
├── tests/                        # Automated tests (optional)
├── .env.example                  # Example environment file for local setup
├── .gitignore                    # Specifies intentionally untracked files
├── Dockerfile                    # Defines the Docker image for the dev environment
├── README.md                     # Main project README
└── Taskfile.yml                  # Go Task definitions for project automation
```

- Create the `.devcontainer`, `.github/workflows`, and `.vscode` directories now.
- The `<APP_NAME>` directory will be created later, potentially by a task.
- The `.devcontainer/custom_setup.sh` file is optional; create it if you intend to manage custom setup steps this way.

### 3.3 `.gitignore`

Create a `.gitignore` file in your project root (`MySplunkProject/`) with the following content:

```
# Environment variables
.env

# VS Code
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.Python
pip-wheel-metadata/
.installed.cfg
*.egg
venv/
.venv/
env/
.env/

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
# package-lock.json # Commit if your team uses npm consistently
# yarn.lock # Commit if your team uses yarn consistently

# Docker
docker-compose.override.yml

# Splunk App specific (local overrides, packaged files not in releases)
**/local/*
!**/local/indexes.conf.example # Example for local index conf
!**/local/inputs.conf.example  # Example for local inputs conf
*.spl
*.tgz

# Build artifacts / Releases (if not managed elsewhere)
releases/
dist/
build/
tmp_downloads/

# OS specific
.DS_Store
Thumbs.db
```

## 4. Dockerized Development Environment (Dev Container)

The dev container provides a consistent Linux environment with Splunk and all necessary development tools pre-installed.

### 4.1 `Dockerfile`

Create a `Dockerfile` in your project root (`MySplunkProject/`).

**Choosing a Base Image:**

You have two main approaches for the base image:

1. **Generic Linux Base (e.g., Ubuntu - Recommended for Dev Sandbox Flexibility):**

   - **Pros:** Full control over the environment, easy to install specific versions of dev tools (Node.js, Python for linters, etc.), Splunk lifecycle managed explicitly via tasks, container stays running (`sleep infinity`) for general dev work even if Splunk is stopped.
   - **Cons:** You are responsible for scripting the entire Splunk installation and setup. Installing apps like MLTK/PSC requires manual scripting (download + install commands).
   - The example below uses this approach.

2. **Official `splunk/splunk` Base:**

   - **Pros:** Uses Splunk's official image with Ansible-based setup. Splunk is pre-installed and configured. Easily install apps like MLTK/PSC on container startup using the `SPLUNK_APPS_URL` environment variable (set in `devcontainer.json`).

   - **Cons:** Splunk runs as the main process (PID 1); if it stops, the container might stop. Less flexibility for a "generic" dev sandbox if you need the container alive without Splunk running. You'd layer dev tools on top of an already opinionated Splunk image.

   - If choosing this, your `Dockerfile` would be much simpler, e.g.:

     ```
     FROM splunk/splunk:latest # Or a specific version like 9.3.0
     # ARG APP_NAME="YourApp" # If using build-time app name
     # ENV APP_NAME=${APP_NAME}
     # Dev tools would be added via devcontainer.json features or RUN commands here if needed
     # USER root # if you need to install tools as root
     # RUN apt-get update && apt-get install -y nodejs npm git # Example
     # USER ${SPLUNK_USER} # Switch back to splunk user
     ```

     You would then set `SPLUNK_APPS_URL`, `SPLUNK_PASSWORD`, etc., in `devcontainer.json`'s `containerEnv`.

**Example `Dockerfile` (Generic Ubuntu Base - Recommended for this Guide):**

```
# Using a recent Ubuntu LTS as a generic base for flexibility
FROM ubuntu:22.04

ARG SPLUNK_VERSION=9.2.1
ARG SPLUNK_BUILD=775318378675
ARG SPLUNK_FILENAME=splunk-${SPLUNK_VERSION}-${SPLUNK_BUILD}-Linux-x86_64.tgz
ARG SPLUNK_DOWNLOAD_URL=https://download.splunk.com/products/splunk/releases/${SPLUNK_VERSION}/linux/${SPLUNK_FILENAME}
# For ARM64/Apple Silicon, ensure you use an ARM-compatible Splunk build and base image.

ENV SPLUNK_HOME=/opt/splunk
ENV SPLUNK_USER=splunk
ENV SPLUNK_GROUP=splunk
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 # Set locale to avoid issues with some tools

# Install dependencies: wget, ca-certs, sudo, python3, pip, git, and other common tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    sudo \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    jq \
    # Add any other system dependencies your app or tools might need (e.g., libffi-dev, gcc for some Python packages)
    && rm -rf /var/lib/apt/lists/*

# Create splunk user and group
RUN groupadd -r ${SPLUNK_GROUP} && \
    useradd -r -m -g ${SPLUNK_GROUP} -d ${SPLUNK_HOME} ${SPLUNK_USER} && \
    # Allow splunk user to use sudo without a password (for dev container convenience if needed)
    # Use with caution and consider if truly necessary for your use case.
    echo "${SPLUNK_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /var/log/splunk && \
    chown -R ${SPLUNK_USER}:${SPLUNK_GROUP} /var/log/splunk

# Download and install Splunk
RUN wget --progress=bar:force -O /tmp/${SPLUNK_FILENAME} ${SPLUNK_DOWNLOAD_URL} && \
    tar -xzf /tmp/${SPLUNK_FILENAME} -C /opt && \
    rm /tmp/${SPLUNK_FILENAME} && \
    chown -R ${SPLUNK_USER}:${SPLUNK_GROUP} ${SPLUNK_HOME}

# Set up Splunk environment, PATH for splunk user
ENV PATH=${SPLUNK_HOME}/bin:${PATH}

# Install Python packages (globally for simplicity in dev container for dev tools)
# These are for the container's general Python environment, not Splunk's internal Python.
# Splunk apps will use Splunk's bundled Python or PSC.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    splunk-sdk \
    pylint \
    autopep8 \
    splunk-packaging-toolkit # For 'slim' command
    # Add other Python tools like 'requests', 'pytest' if needed for dev scripts

# Create workspace directory and set permissions
RUN mkdir -p /workspace && \
    chown -R ${SPLUNK_USER}:${SPLUNK_GROUP} /workspace

# Splunk license and first-time run handling
# The SPLUNK_PASSWORD env var (from devcontainer.json) will be used by tasks or scripts
# when starting Splunk for the first time.
ENV SPLUNK_START_ARGS="--accept-license --answer-yes --no-prompt"
# Default SPLUNK_PASSWORD, will be overridden by devcontainer.json or .env
ENV SPLUNK_PASSWORD="changeme"

# Expose Splunk ports
EXPOSE 8000 8089

USER ${SPLUNK_USER}
WORKDIR ${SPLUNK_HOME}

# CMD to keep the container running for dev purposes. Splunk is started via tasks.
CMD ["sleep", "infinity"]
```

### 4.2 Dev Container Configuration (`.devcontainer/devcontainer.json`)

Create `.devcontainer/devcontainer.json`. The `containerEnv` section now defaults `INSTALL_MLTK` and `INSTALL_PSC` to `true`.

```
{
  "name": "Splunk App Development Environment",
  "build": {
    "dockerfile": "../Dockerfile",
    "context": ".."
  },
  "runArgs": [
    "--cap-add=IPC_LOCK"
  ],
  "workspaceFolder": "/workspace",
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=cached",

  "features": {
    "ghcr.io/devcontainers-contrib/features/task:1": {},
    "ghcr.io/devcontainers/features/git:1": {},
    "ghcr.io/devcontainers/features/node:1": {
      "version": "lts"
    }
  },

  "customizations": {
    "vscode": {
      "extensions": [
        "splunk.vscode-extension-splunk",
        "ms-python.python",
        "redhat.vscode-yaml",
        "bierner.markdown-preview-github-styles",
        "mutantdino.resourcemonitor"
      ],
      "settings": {
        "splunk.path.splunkHome": "/opt/splunk",
        "splunk.path.splunkExe": "/opt/splunk/bin/splunk",
        "python.defaultInterpreterPath": "/usr/bin/python3",
        "files.eol": "\n",
        "python.linting.pylintEnabled": true,
        "python.linting.enabled": true,
        "python.formatting.provider": "autopep8",
        "[python]": {
          "editor.defaultFormatter": "ms-python.autopep8"
        },
        "[yaml]": {
          "editor.defaultFormatter": "redhat.vscode-yaml"
        }
      }
    }
  },

  "postCreateCommand": "bash .devcontainer/post-create.sh",

  "forwardPorts": [
    8000,
    8089,
    3000
  ],

  "remoteUser": "splunk",

  "containerEnv": {
    "APP_NAME": "${localEnv:APP_NAME_DEVCONTAINER:MyDefaultSplunkApp}",
    "SPLUNKBASE_USERNAME": "${localEnv:SPLUNKBASE_USERNAME}",
    "SPLUNKBASE_PASSWORD": "${localEnv:SPLUNKBASE_PASSWORD}",
    "SPLUNK_PASSWORD": "${localEnv:SPLUNK_ADMIN_PASSWORD_DEVCONTAINER:changeme}",
    "NODE_ENV": "development",
    "INSTALL_MLTK": "${localEnv:INSTALL_MLTK:true}",  # Default to install
    "MLTK_APP_URL": "${localEnv:MLTK_APP_URL:https://splunkbase.splunk.com/app/2890/release/5.6.0/download}",
    "INSTALL_PSC": "${localEnv:INSTALL_PSC:true}",    # Default to install
    "PSC_APP_URL": "${localEnv:PSC_APP_URL:https://splunkbase.splunk.com/app/2882/release/4.2.3/download}"
  }
}
```

### 4.3 User-Defined Custom Setup Script (`.devcontainer/custom_setup.sh`)

Create an **optional** script named `.devcontainer/custom_setup.sh`. If this file exists, `post-create.sh` will source it to perform additional user-defined setup steps, such as installing Splunk apps like MLTK or PSC. The comments reflect the new default behavior.

```
#!/bin/bash
# .devcontainer/custom_setup.sh
# This script is sourced by post-create.sh if it exists.
# It can call functions defined in post-create.sh (e.g., 'install_splunk_app_from_url')
# and perform other custom setup tasks.
# MLTK and PSC installation are now defaulted to 'true' in devcontainer.json.
# Users can set INSTALL_MLTK=false or INSTALL_PSC=false in their .env file to prevent installation.

echo "--- Running .devcontainer/custom_setup.sh for user-defined configurations ---"

# --- Example: Splunk Machine Learning Toolkit (MLTK) Installation ---
# Controlled by INSTALL_MLTK and MLTK_APP_URL environment variables.
# Default is to install (INSTALL_MLTK:true in devcontainer.json).
# User can set INSTALL_MLTK=false in .env to disable.
if [ "$INSTALL_MLTK" = "true" ] && [ -n "$MLTK_APP_URL" ]; then
  echo "MLTK installation requested (or defaulted to true)."
  # Calls the function defined in post-create.sh
  install_splunk_app_from_url "MLTK" "$MLTK_APP_URL"
else
  echo "Skipping MLTK installation (INSTALL_MLTK is 'false' or MLTK_APP_URL is missing/empty)."
fi

# --- Example: Python for Scientific Computing (PSC) Installation ---
# Controlled by INSTALL_PSC and PSC_APP_URL environment variables.
# Default is to install (INSTALL_PSC:true in devcontainer.json).
# User can set INSTALL_PSC=false in .env to disable.
if [ "$INSTALL_PSC" = "true" ] && [ -n "$PSC_APP_URL" ]; then
  echo "PSC installation requested (or defaulted to true)."
  install_splunk_app_from_url "PSC" "$PSC_APP_URL"
else
  echo "Skipping PSC installation (INSTALL_PSC is 'false' or PSC_APP_URL is missing/empty)."
fi

# --- Example: Manual Toggling for App Installations (Alternative User Control) ---
# If you prefer to always control installations by editing this script directly (ignoring .env flags for these specific apps):
# 1. Set INSTALL_MLTK=false and INSTALL_PSC=false in devcontainer.json or your .env to avoid double installs.
# 2. Uncomment the lines below to force manual installation.

# Example for MLTK
# echo "Manual MLTK installation attempt (from custom_setup.sh)..."
# mltk_manual_url="https://splunkbase.splunk.com/app/2890/release/5.6.0/download" # Replace with desired version
# install_splunk_app_from_url "MLTK (Manual)" "$mltk_manual_url"

# Example for PSC
# echo "Manual PSC installation attempt (from custom_setup.sh)..."
# psc_manual_url="https://splunkbase.splunk.com/app/2882/release/4.2.3/download" # Replace with desired version
# install_splunk_app_from_url "PSC (Manual)" "$psc_manual_url"


# --- Add other custom app installations or setup commands here ---
# For example, installing another Splunk app:
# echo "Installing MyCustomTA..."
# custom_ta_url="https://splunkbase.example.com/app/1234/release/1.0.0/download"
# install_splunk_app_from_url "MyCustomTA" "$custom_ta_url"

# Or, run other setup commands:
# echo "Performing other custom setup tasks..."
# mkdir -p /workspace/my_custom_data_dir
# touch /workspace/my_custom_flag_file.txt

echo "--- Finished .devcontainer/custom_setup.sh ---"
```

Make this script executable: `chmod +x .devcontainer/custom_setup.sh`.

### 4.4 Main Post-Creation Script (`.devcontainer/post-create.sh`)

Modify .devcontainer/post-create.sh. It defines the install_splunk_app_from_url function and sources the custom_setup.sh script if it exists.

(No logical changes here, just ensuring comments align if any were specific to the old default)

```
#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Running post-create script as user: $(whoami)"
echo "SPLUNK_HOME is: $SPLUNK_HOME" # Verify SPLUNK_HOME is available

# --- Helper Function to Install Splunk Apps ---
# This function can be called by the sourced custom_setup.sh script
install_splunk_app_from_url() {
  local app_name_simple="$1" # e.g., "MLTK"
  local app_url="$2"
  # Splunkbase download URLs might not have .spl in them and might redirect.
  # We'll use a generic temporary filename for the download.
  local temp_spl_filename="${app_name_simple}_downloaded_package.spl" 
  local download_dir="/tmp/splunk_app_downloads_$$" # Use PID for unique temp dir

  echo "Attempting to install $app_name_simple from $app_url..."
  mkdir -p "$download_dir"

  echo "Downloading $app_name_simple..."
  # Use curl with -L to follow redirects and -o to save to a file.
  # Adding user-agent as some servers might require it.
  if curl --user-agent "DevContainer Setup Script/1.0" --silent --show-error --fail -L "$app_url" -o "$download_dir/$temp_spl_filename"; then
    echo "Downloaded $app_name_simple to $download_dir/$temp_spl_filename successfully."

    echo "INSTALLING $app_name_simple: Unpacking $download_dir/$temp_spl_filename to $SPLUNK_HOME/etc/apps/"
    mkdir -p "$SPLUNK_HOME/etc/apps/" # Ensure target directory exists
    
    # Unpack (.spl files are typically .tar.gz)
    if tar -xzf "$download_dir/$temp_spl_filename" -C "$SPLUNK_HOME/etc/apps/"; then
      echo "$app_name_simple unpacked successfully. Restart Splunk to enable."
    else
      echo "ERROR: Failed to unpack $temp_spl_filename for $app_name_simple."
    fi
    rm "$download_dir/$temp_spl_filename"
  else
    echo "ERROR: Failed to download $app_name_simple from $app_url. Check URL and network. HTTP Status might provide clues if download fails."
  fi
  rm -rf "$download_dir" # Clean up temp dir
}
export -f install_splunk_app_from_url # Export the function so sourced scripts can use it

# --- Python Development Dependencies ---
if [ -f "/workspace/requirements-dev.txt" ]; then
  echo "Installing Python development dependencies from /workspace/requirements-dev.txt..."
  python3 -m pip install --no-cache-dir -r /workspace/requirements-dev.txt
else
  echo "No /workspace/requirements-dev.txt found. Core Python dev tools assumed installed by Dockerfile."
fi

# --- Node.js & Yeoman for Splunk App Scaffolding ---
if command -v npm &> /dev/null; then
  echo "Node.js found. Installing/updating Yeoman and generator-splunk-app globally..."
  npm install -g yo generator-splunk-app
  echo "Yeoman and generator-splunk-app installed."
else
  echo "Node.js (npm) not found. Skipping installation of Yeoman and generator-splunk-app."
fi

# --- Source Optional User Script for Custom Setup (App Installations, etc.) ---
USER_CUSTOM_SETUP_SCRIPT=".devcontainer/custom_setup.sh"
if [ -f "$USER_CUSTOM_SETUP_SCRIPT" ]; then
  echo "Found $USER_CUSTOM_SETUP_SCRIPT. Sourcing it for additional setup..."
  # Sourcing the script executes it in the current shell context,
  # allowing it to use functions defined above (like install_splunk_app_from_url)
  # and modify the current environment if needed.
  . "$USER_CUSTOM_SETUP_SCRIPT"
else
  echo "$USER_CUSTOM_SETUP_SCRIPT not found. Skipping user-defined custom setup."
fi
echo "Note: If any Splunk apps were installed via custom_setup.sh, you might need to run 'task splunk:restart' after the container is fully up and Splunk has started once."


# --- Git Configuration (Optional) ---
# Example:
# if [ -n "${GIT_USER_NAME_DEVCONTAINER}" ] && [ -n "${GIT_USER_EMAIL_DEVCONTAINER}" ]; then
#   git config --global user.name "${GIT_USER_NAME_DEVCONTAINER}"
#   git config --global user.email "${GIT_USER_EMAIL_DEVCONTAINER}"
#   echo "Git user configured in container as: ${GIT_USER_NAME_DEVCONTAINER} <${GIT_USER_EMAIL_DEVCONTAINER}>"
# fi

echo "-----------------------------------------------------"
echo "Post-create script finished."
echo "VS Code Terminal is connected to the dev container."
echo "Splunk App Name for tasks: ${APP_NAME}" # APP_NAME is from containerEnv
echo "To create a Splunk app: task app:create"
echo "To start Splunk: task splunk:start"
echo "-----------------------------------------------------"

# Output versions of key tools for verification
echo "Go Task version: $(task --version)"
if command -v node &> /dev/null; then echo "Node version: $(node --version)"; fi
if command -v npm &> /dev/null; then echo "npm version: $(npm --version)"; fi
if command -v python3 &> /dev/null; then echo "Python3 version: $(python3 --version)"; fi
if command -v python3 -m pip &> /dev/null; then echo "pip3 version: $(python3 -m pip --version)"; fi
if command -v yo &> /dev/null; then echo "Yeoman (yo) version: $(yo --version)"; fi
if command -v slim &> /dev/null; then echo "slim (Splunk Packaging Toolkit) version: $(slim --version)"; fi
```

Make `post-create.sh` executable: `chmod +x .devcontainer/post-create.sh`.

### 4.5 Environment Variables (`.env` and `.env.example`)

Update `.env.example` in the project root. The comments now clarify that MLTK/PSC installation is *on by default* and how to disable it.

```
# .env.example - Copy to .env and fill in your values. DO NOT COMMIT .env

# Name for your Splunk App directory and used by various tasks/scripts
APP_NAME_DEVCONTAINER=MyCoolSplunkApp

# Your Splunkbase credentials (if needed by scripts for app downloads/updates OR if using splunk/splunk base with SPLUNK_APPS_URL for private apps)
SPLUNKBASE_USERNAME=your_splunkbase_username
SPLUNKBASE_PASSWORD=your_splunkbase_password_secret

# Default Splunk admin password for the containerized instance
SPLUNK_ADMIN_PASSWORD_DEVCONTAINER=SecureSplunkDevPassword123!

# --- Optional App Installations (MLTK, PSC) via .devcontainer/custom_setup.sh ---
# By default, MLTK and PSC WILL BE INSTALLED (as per devcontainer.json defaults).
# To PREVENT installation, uncomment and set the corresponding variable to 'false'.
# INSTALL_MLTK=false
# INSTALL_PSC=false

# The URLs below are used if installation is enabled. You can override them here for different versions.
MLTK_APP_URL=https://splunkbase.splunk.com/app/2890/release/5.6.0/download
PSC_APP_URL=https://splunkbase.splunk.com/app/2882/release/4.2.3/download

# Optional: Git user details for commits made from within the container
# GIT_USER_NAME_DEVCONTAINER="Your Name"
# GIT_USER_EMAIL_DEVCONTAINER="your.email@example.com"
```

## 5. Task Automation with Go Task

Go Task provides simple command aliases for common development operations.

### 5.1 `Taskfile.yml`

Create `Taskfile.yml` in the project root:

```
version: '3'

vars:
  SPLUNK_HOME: /opt/splunk
  # APP_NAME is sourced from the environment variable set in devcontainer.json
  APP_NAME_ENV: '{{env "APP_NAME"}}'
  # Default app name if APP_NAME env var is not set or empty.
  APP_NAME: '{{if .APP_NAME_ENV}}{{.APP_NAME_ENV}}{{else}}MyDefaultSplunkApp{{end}}'
  APP_DIR: "/workspace/{{.APP_NAME}}" # Path to the app within the workspace

tasks:
  default:
    desc: "List all available tasks"
    cmds:
      - task --list-all
    silent: true

  # --- Splunk Instance Management ---
  splunk:start:
    desc: "Start Splunk instance (accepts license, uses SPLUNK_PASSWORD from env)"
    cmds:
      # Ensure SPLUNK_PASSWORD is set in the container's environment
      # The SPLUNK_START_ARGS are also in the Dockerfile env, but explicit here is fine.
      - "{{.SPLUNK_HOME}}/bin/splunk start --accept-license --answer-yes --no-prompt"
    # Add pre-conditions if necessary, e.g., check if already running

  splunk:stop:
    desc: "Stop Splunk instance"
    cmds:
      - "{{.SPLUNK_HOME}}/bin/splunk stop"

  splunk:restart:
    desc: "Restart Splunk instance"
    cmds:
      - "{{.SPLUNK_HOME}}/bin/splunk restart"

  splunk:status:
    desc: "Check Splunk status"
    cmds:
      - "{{.SPLUNK_HOME}}/bin/splunk status"

  splunk:logs:
    desc: "Tail Splunkd logs (splunkd.log)"
    cmds:
      - tail -F {{.SPLUNK_HOME}}/var/log/splunk/splunkd.log

  splunk:web-logs:
    desc: "Tail Splunk Web logs (web_service.log)"
    cmds:
      - tail -F {{.SPLUNK_HOME}}/var/log/splunk/web_service.log

  # --- Splunk App Development ---
  app:create:
    desc: "Create a new Splunk app from template using Yeoman generator (yo splunk)"
    summary: |
      Scaffolds a new Splunk app in /workspace using 'yo splunk'.
      The app will be created in a directory named by the APP_NAME environment variable (default: {{.APP_NAME}}).
      If the directory /workspace/{{.APP_NAME}} already exists, it will not overwrite.
      You can choose to create a React-based app during the Yeoman prompts.
    cmds:
      - |
        if [ -z "$APP_NAME" ]; then # Check the shell variable, which Task sets from vars
          echo "Error: APP_NAME is not set or is empty. Check Taskfile.yml vars and containerEnv."
          exit 1
        fi
        if [ -d "{{.APP_DIR}}" ]; then
          echo "App directory '{{.APP_DIR}}' already exists. Skipping creation."
        else
          echo "Creating Splunk app named '{{.APP_NAME}}' in '{{.APP_DIR}}'..."
          mkdir -p /workspace # Ensure workspace exists
          cd /workspace
          # 'yo splunk' will prompt for app details.
          # Pass APP_NAME as an argument to 'yo splunk' if it accepts it to pre-fill or name the directory.
          # Most Yeoman generators will create a subdir based on the name you provide in prompts.
          yo splunk "{{.APP_NAME}}"
          echo "Splunk app '{{.APP_NAME}}' scaffolding initiated."
          echo "If 'yo splunk' created a differently named directory, please rename it to '{{.APP_NAME}}' or update your APP_NAME env var."
          echo "Please review the generated files and customize as needed."
        fi
    preconds:
      - sh: "command -v yo"
        msg: "Yeoman (yo) command not found. Ensure Node.js feature is enabled and post-create script ran."
      - sh: "npm list -g generator-splunk-app | grep generator-splunk-app"
        msg: "generator-splunk-app not found globally. Check post-create script."

  app:package:
    desc: "Package the Splunk app (using 'slim package')"
    dir: "{{.APP_DIR}}" # Run this task from within the app directory
    cmds:
      - echo "Packaging app '{{.APP_NAME}}' from '{{.APP_DIR}}'..."
      - |
        if [ ! -d "." ]; then # Checks if the current directory (APP_DIR) exists
          echo "Error: App directory '{{.APP_DIR}}' does not exist or task is not run from within it."
          exit 1
        fi
        mkdir -p ../releases # Create releases dir at workspace level
        # slim package -o <output_dir_relative_to_app_dir> <source_dir_relative_to_app_dir>
        slim package -o ../releases .
        echo "App packaged. Check the /workspace/releases directory."
    preconds:
      - sh: "[ -d '{{.APP_DIR}}' ]"
        msg: "App directory '{{.APP_DIR}}' not found. Create the app first (task app:create)."
      - sh: "command -v slim"
        msg: "slim (Splunk Packaging Toolkit) not found. Ensure it's installed (e.g., via pip in Dockerfile or requirements-dev.txt)."

  app:install-local:
    desc: "Install/copy the current app to the local Splunk instance's apps directory"
    cmds:
      - |
        if [ ! -d "{{.APP_DIR}}" ]; then
          echo "Error: App source directory '{{.APP_DIR}}' not found."
          exit 1
        fi
        echo "Copying '{{.APP_DIR}}' to '{{.SPLUNK_HOME}}/etc/apps/'..."
        mkdir -p {{.SPLUNK_HOME}}/etc/apps/
        # Consider removing existing app dir in Splunk to avoid merge issues (optional)
        # rm -rf "{{.SPLUNK_HOME}}/etc/apps/{{.APP_NAME}}"
        cp -R "{{.APP_DIR}}" "{{.SPLUNK_HOME}}/etc/apps/"
        echo "App '{{.APP_NAME}}' copied to Splunk. Restart Splunk to apply changes (task splunk:restart)."

  # --- Python Development ---
  python:lint:
    desc: "Lint Python code in the app using Pylint"
    dir: "{{.APP_DIR}}"
    cmds:
      - |
        if [ ! -d "." ]; then echo "Error: App directory '{{.APP_DIR}}' not found."; exit 1; fi
        echo "Linting Python files in {{.APP_DIR}}..."
        # Adjust path if Python code is in specific subdirs like 'bin' or 'lib'
        pylint **/*.py || true # Allow to continue even if linting errors
    preconds:
      - sh: "command -v pylint"
        msg: "pylint not found. Install via pip (e.g., in Dockerfile or requirements-dev.txt)."

  python:format:
    desc: "Format Python code in the app using autopep8"
    dir: "{{.APP_DIR}}"
    cmds:
      - |
        if [ ! -d "." ]; then echo "Error: App directory '{{.APP_DIR}}' not found."; exit 1; fi
        echo "Formatting Python files in {{.APP_DIR}}..."
        autopep8 --in-place --recursive .
    preconds:
      - sh: "command -v autopep8"
        msg: "autopep8 not found. Install via pip."
```

### 5.2 Using Tasks

Once the dev container is running:

1. Open the VS Code Command Palette (Ctrl+Shift+P or Cmd+Shift+P).
2. Type "Tasks: Run Task".
3. Select "task" (it might show sub-options if other task providers are present).
4. Choose the desired task from the list (e.g., `app:create`, `splunk:start`).

## 6. CI/CD with GitHub Actions

Automate building, testing, packaging, and releasing your Splunk app.

### 6.1 Workflow File (`.github/workflows/release.yml`)

Create .github/workflows/release.yml with the following content.

Remember to customize the APP_NAME environment variable at the top of this YAML file to match your Splunk app's directory name.

```
# .github/workflows/release.yml
name: Splunk App CI/CD

on:
  push:
    branches:
      - main # Or your primary development branch
      - master
  release:
    types: [published] # Triggers when a new release is published in GitHub

env:
  # Define your Splunk app's name here.
  # This should match the directory name of your app and often the ID in app.conf.
  # For consistency, you might want this to be the same as APP_NAME_DEVCONTAINER in your .env
  APP_NAME: MyCoolSplunkApp # <-- !!! IMPORTANT: SET YOUR APP NAME HERE !!!
  # You can also try to derive this from app.conf if needed, but explicit is often easier.

jobs:
  build_and_validate:
    name: Build and Validate App
    runs-on: ubuntu-latest
    outputs:
      app_package_name: ${{ steps.package_app.outputs.app_package_name }}
      app_package_path: ${{ steps.package_app.outputs.app_package_path }}
      # appinspect_report_name: ${{ steps.validate_app.outputs.appinspect_report_name }} # From official action
      # appinspect_report_path: ${{ steps.validate_app.outputs.appinspect_report_path }} # From official action

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x' # Or a specific version like '3.9'

      - name: Install Splunk Packaging Toolkit (slim) and AppInspect
        run: |
          python -m pip install --upgrade pip
          pip install splunk-packaging-toolkit splunk-appinspect

      - name: Package Splunk App
        id: package_app
        run: |
          if [ ! -d "${{ env.APP_NAME }}" ]; then
            echo "Error: App directory '${{ env.APP_NAME }}' not found at the repository root."
            exit 1
          fi
          cd "${{ env.APP_NAME }}"
          mkdir -p ../releases
          slim package -o ../releases .
          
          PACKAGE_FILE=$(find ../releases -maxdepth 1 \( -name "*.spl" -o -name "*.tgz" \) -print -quit)

          if [ -z "$PACKAGE_FILE" ]; then
            echo "Error: No .spl or .tgz package file found in ../releases after running 'slim package'."
            ls -l ../releases
            exit 1
          fi
          
          echo "Packaged app: $PACKAGE_FILE"
          echo "app_package_name=$(basename "$PACKAGE_FILE")" >> $GITHUB_OUTPUT
          # GITHUB_OUTPUT needs absolute path or path relative to GITHUB_WORKSPACE
          echo "app_package_path=${GITHUB_WORKSPACE}/releases/$(basename "$PACKAGE_FILE")" >> $GITHUB_OUTPUT
        shell: bash

      - name: Validate App with AppInspect
        id: validate_app
        uses: splunk/splunk-app-action-appinspect@v1
        with:
          app_package_path: ${{ steps.package_app.outputs.app_package_path }}
          output_file: "appinspect_report.json" # Name of the report file

      - name: Upload AppInspect Report as Artifact
        uses: actions/upload-artifact@v4
        if: always() 
        with:
          name: appinspect-report
          path: ${{ steps.validate_app.outputs.output_file }} 

      - name: Upload Packaged App as Artifact (for other jobs)
        uses: actions/upload-artifact@v4
        with:
          name: splunk-app-package
          # Path needs to be relative to GITHUB_WORKSPACE or absolute
          path: ${{ steps.package_app.outputs.app_package_path }}

  create_github_release:
    name: Create GitHub Release
    if: github.event_name == 'release' && github.event.action == 'published'
    needs: build_and_validate
    runs-on: ubuntu-latest
    permissions:
      contents: write 

    steps:
      - name: Download Packaged App Artifact
        uses: actions/download-artifact@v4
        with:
          name: splunk-app-package
          path: ./release_package 

      - name: Download AppInspect Report Artifact
        uses: actions/download-artifact@v4
        with:
          name: appinspect-report
          path: ./appinspect_report

      - name: Create Release and Upload Assets
        uses: softprops/action-gh-release@v2
        with:
          files: |
            ./release_package/*.* ./appinspect_report/*.*

  publish_to_splunkbase:
    name: Publish to Splunkbase
    if: github.event_name == 'release' && github.event.action == 'published' && !github.event.release.prerelease
    needs: build_and_validate 
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code 
        uses: actions/checkout@v4

      - name: Download Packaged App Artifact
        uses: actions/download-artifact@v4
        with:
          name: splunk-app-package
          path: ./release_package 

      - name: Get App ID from app.conf
        id: get_app_id
        run: |
          APP_ID_FROM_CONF=$(awk -F' *= *' '/^id *=[^#]*/ {print $2; exit}' "${{ env.APP_NAME }}/default/app.conf")
          if [ -z "$APP_ID_FROM_CONF" ]; then
            echo "Error: Could not extract 'id' from ${{ env.APP_NAME }}/default/app.conf"
            exit 1
          fi
          echo "App ID from app.conf: $APP_ID_FROM_CONF"
          echo "app_id=$APP_ID_FROM_CONF" >> $GITHUB_OUTPUT
        shell: bash

      - name: Publish to Splunkbase
        uses: splunk/splunk-app-action-publish@v1 
        with:
          app_id: ${{ steps.get_app_id.outputs.app_id }}
          # The path to the package needs to be correct after download
          app_package_path: "./release_package/${{ needs.build_and_validate.outputs.app_package_name }}"
          splunkbase_username: ${{ secrets.SPLUNKBASE_USERNAME }}
          splunkbase_password: ${{ secrets.SPLUNKBASE_PASSWORD }}
```

### 6.2 Setting up GitHub Secrets

For the `publish_to_splunkbase` job to work, you need to add your Splunkbase credentials as secrets in your GitHub repository:

1. Go to your GitHub repository > Settings > Secrets and variables > Actions.
2. Click "New repository secret".
3. Create two secrets:
   - `SPLUNKBASE_USERNAME`: Your Splunkbase username.
   - `SPLUNKBASE_PASSWORD`: Your Splunkbase password.

## 7. Development Workflow

1. **Open Project in Dev Container:**
   - Open your project folder (`MySplunkProject`) in VS Code.
   - If prompted, click "Reopen in Container". If not prompted, click the green icon in the bottom-left corner of VS Code and select "Reopen in Container".
   - The first time, Docker will build the image, and the `post-create.sh` script (and `custom_setup.sh` if it exists) will run. This may take several minutes. Review its output for app installation status (MLTK and PSC should install by default unless disabled in `.env`).
2. **Accessing the Terminal:**
   - Once the container is running, open a new terminal in VS Code (Ctrl+` or Cmd+`). This terminal is running *inside* the dev container.
3. **Create Your App (if it doesn't exist):**
   - Run the Go Task: `task app:create`. This will use Yeoman to scaffold a new Splunk app in `/workspace/YourAppName` (where `YourAppName` is determined by the `APP_NAME` environment variable). Follow the Yeoman prompts (you can choose a React-based app here).
4. **Start Splunk:**
   - Run `task splunk:start`.
   - Access Splunk Web at `http://localhost:8000`. Log in with `admin` and the password you set via `SPLUNK_ADMIN_PASSWORD_DEVCONTAINER` in your `.env` file.
   - Verify MLTK/PSC (if their installation was enabled/defaulted to true) appear in the "Apps" list. You might need to run `task splunk:restart` if they don't show up immediately after the first start.
5. **Develop Your App:**
   - Modify files in the `/workspace/YourAppName` directory. These changes are reflected directly from your local file system into the container.
   - Use other tasks like `python:lint`, `python:format`, `app:install-local` (to copy your app to Splunk's `etc/apps`), and `splunk:restart` as needed.
6. **Commit and Push Changes:**
   - Use Git as usual from your local machine or from within the VS Code UI (which can use Git from the container if configured).
7. **CI/CD Pipeline:**
   - Pushing to `main`/`master` or creating a release on GitHub will trigger the GitHub Actions workflow.

## 8. Conclusion & Next Steps

This setup provides a comprehensive, automated, and consistent environment for Splunk app development.

**Next Steps & Considerations:**

- **Customize Dockerfile:** Add any specific system libraries or tools your app requires.
- **Refine `Taskfile.yml`:** Add more tasks specific to your workflow (e.g., running specific tests, deploying to a dev Splunk instance).
- **Enhance `custom_setup.sh`:** Add more setup steps, like pre-configuring Splunk settings or installing other specific app dependencies beyond MLTK/PSC if needed.
- **Testing Strategy:** Integrate automated testing (unit, integration) into your `Taskfile.yml` and GitHub Actions workflow.
- **Splunk AppInspect Tags:** Customize the `included_tags` and `excluded_tags` in the `Validate App with AppInspect` step of your GitHub Actions workflow based on your app's target environment (e.g., Splunk Cloud, on-premises).
- **Semantic Versioning & Changelog:** Implement a process for versioning your app and maintaining a changelog, which can be used in the GitHub Release notes.
- **MLTK/PSC URLs:** The example URLs in `devcontainer.json` and `.env.example` for MLTK/PSC are for specific versions. You will need to find and verify the correct direct download links for the versions you intend to use for them to be successfully downloaded and installed by the `custom_setup.sh` script.

This guide should give you a strong foundation for efficient Splunk app development.