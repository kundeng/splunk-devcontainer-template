#!/bin/bash
set -e

echo "=== Splunk Dev: post-create setup ==="

# ── PATH setup ───────────────────────────────────────────────────────
mkdir -p "$HOME/.local/bin"
if ! grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null; then
  # shellcheck disable=SC2016
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
export PATH="$HOME/.local/bin:$PATH"

# ── Python dev tools ─────────────────────────────────────────────────
echo "Installing Python tools (appinspect, ruff, pytest)..."
pip install --user --quiet splunk-appinspect ruff pytest 2>/dev/null

# ── Create .env from example if missing ──────────────────────────────
if [ ! -f /workspace/.env ] && [ -f /workspace/splunk.env.example ]; then
  cp /workspace/splunk.env.example /workspace/.env
  echo ".env created from splunk.env.example"
fi

# ── Ensure directory structure ───────────────────────────────────────
mkdir -p /workspace/splunk/config/apps
mkdir -p /workspace/splunk/stage
mkdir -p /workspace/packages

# ── Build Splunk image ────────────────────────────────────────────────
echo "Building Splunk dev image (first time only)..."
task dev:build-image 2>&1 | tail -5 || true

echo "=== Setup complete. Run 'task dev:up' to start Splunk. ==="
