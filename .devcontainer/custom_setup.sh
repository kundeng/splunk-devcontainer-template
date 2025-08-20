#!/bin/bash
# This script is sourced by post-create.sh and can be customized for specific setup needs

echo "Running custom setup script..."

# --- Install app dependencies (MLTK, PSC) ---
# This uses the app:install-deps task from Taskfile.yml
# Controlled by environment variables in .env:
# - INSTALL_MLTK=true|false
# - MLTK_APP_URL=url
# - INSTALL_PSC=true|false
# - PSC_APP_URL=url
echo "Installing app dependencies..."
task app:install-deps

# --- Add other custom app installations or setup commands here ---
# For example, installing another Splunk app from Splunkbase:
# task app:download-from-splunkbase -- 1234-1.0.0

# For example, installing another Splunk app from GitHub:
# task app:download-from-github -- https://api.github.com/repos/user/repo/releases/tags/v1.0.0

# --- Example: Additional Python packages ---
# echo "Installing additional Python packages..."
# pip3 install pandas scikit-learn matplotlib

# --- Example: Additional Node.js packages ---
# echo "Installing additional Node.js packages..."
# npm install -g typescript webpack

# --- Example: Create a new app ---
# task app:create APP_NAME=my_custom_app

# --- Example: Install all apps to Splunk ---
# task app:install-all

echo "Custom setup completed."
