#!/bin/bash
set -e

echo "Running post-create script as user: $(whoami)"

# --- Install development tools ---
echo "Installing development tools..."
npm install -g yo generator-splunk-app @splunk/create

# --- Ensure ~/.local/bin is on PATH persistently ---
mkdir -p "$HOME/.local/bin"
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.profile" 2>/dev/null; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
fi
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
export PATH="$HOME/.local/bin:$PATH"

# --- Go Task is installed via Dev Container Feature ---
echo "Go Task will be provided by devcontainer feature (devcontainers-extra/go-task)."

# --- Install Python packages for AppInspect, Slim, and development ---
echo "Installing Python development packages (AppInspect, Slim, tooling)..."
pip install --user splunk-appinspect splunk-packaging-toolkit black flake8 isort pytest
~/.local/bin/slim --version || true

# --- Create .env file if it doesn't exist ---
echo "Setting up environment variables (.env)..."
if [ ! -f "/workspace/.env" ]; then
  if [ -f "/workspace/splunk.env.example" ]; then
    cp /workspace/splunk.env.example /workspace/.env
    echo ".env created from splunk.env.example"
  else
    cat > /workspace/.env << EOF
SPLUNK_PASSWORD=admin123
SPLUNKBASE_USERNAME=
SPLUNKBASE_PASSWORD=
SPLUNK_APPS_URL=
EOF
    echo ".env created with default values"
  fi
else
  echo ".env already exists"
fi

# --- Initialize workspace structure ---
echo "Initializing workspace structure..."
mkdir -p /workspace/splunk/config
mkdir -p /workspace/splunk/deps
mkdir -p /workspace/splunk/packages
echo "Workspace structure initialized"

echo "Post-create setup completed."
echo "Next steps:"
echo "  1. task splunk:up          # Start Splunk runtime"
echo "  2. task react:create       # Create React app (modern)"
echo "  3. task app:package        # Package apps (legacy tar)"
echo "  4. task app:validate       # Validate with AppInspect"
echo "  5. ~/.local/bin/slim --help  # Splunk Packaging Toolkit (Slim)"
echo ""
echo "Use 'task' to see available development tasks."

# --- Optional custom setup hook ---
if [ -f "/workspace/.devcontainer/custom_setup.sh" ]; then
  echo "Running optional custom_setup.sh..."
  # shellcheck disable=SC1091
  source /workspace/.devcontainer/custom_setup.sh || true
fi

