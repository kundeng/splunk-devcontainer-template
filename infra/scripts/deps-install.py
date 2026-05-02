#!/usr/bin/env python3
"""Install or download Splunkbase dependencies from a deps.yml file.

Modes:
  install (default): Download and install deps into a running Splunk container
  download-only:     Download app tarballs to a local directory (no container needed)

Usage:
    # Install mode (default)
    python3 deps-install.py [--env-file .env] [--deps-file splunk/config/deps.yml]
                            [--stage-dir splunk/stage] [--container splunk-dev]

    # Download-only mode (no container required)
    python3 deps-install.py --download-only --app-id 7404 --version 3.4.1
                            [--env-file .env] [--output-dir ./downloads]

    # Download-only from deps.yml
    python3 deps-install.py --download-only [--deps-file splunk/config/deps.yml]
                            [--output-dir ./downloads]

Credentials resolved in order:
  1. --username / --password CLI args
  2. SPLUNKBASE_USERNAME / SPLUNKBASE_PASSWORD in .env file
  3. SPLUNKBASE_USERNAME / SPLUNKBASE_PASSWORD environment variables
  4. Built-in fallback account

Supports two install methods:
  - splunkbase_id + version: Splunk REST API installs directly from Splunkbase
    (same approach as splunk-ansible — Splunk downloads the app itself)
  - url: download tarball, then install via splunk CLI

No external dependencies — stdlib only (Python 3.8+).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import ssl
import time


# ── Config ───────────────────────────────────────────────────────────

SPLUNKBASE_LOGIN_URL = "https://splunkbase.splunk.com/api/account:login/"
SPLUNK_API_BASE = "https://localhost:8089"
FALLBACK_USERNAME = "bayeslearner@outlook.com"
FALLBACK_PASSWORD = "Welcome1!"


# ── Helpers ──────────────────────────────────────────────────────────

def load_env(env_file: str) -> dict:
    """Parse a .env file into a dict (simple key=value, no interpolation)."""
    env = {}
    if not os.path.isfile(env_file):
        return env
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env


def resolve_credentials(args, env: dict) -> tuple:
    """Resolve Splunkbase credentials: CLI args > .env > env vars > fallback."""
    username = (
        getattr(args, "username", None)
        or env.get("SPLUNKBASE_USERNAME")
        or os.environ.get("SPLUNKBASE_USERNAME")
        or FALLBACK_USERNAME
    )
    password = (
        getattr(args, "password", None)
        or env.get("SPLUNKBASE_PASSWORD")
        or os.environ.get("SPLUNKBASE_PASSWORD")
        or FALLBACK_PASSWORD
    )
    return username, password


def parse_deps(deps_file: str) -> list:
    """Parse deps.yml into a list of dependency dicts."""
    deps = []
    if not os.path.isfile(deps_file):
        print(f"No deps file at {deps_file}")
        return deps

    current = None
    in_deps = False

    with open(deps_file) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "dependencies:":
                in_deps = True
                continue
            if not in_deps:
                continue

            # New dependency item
            m = re.match(r"^\s*-\s*name:\s*(.+)", line)
            if m:
                if current and current.get("name"):
                    deps.append(current)
                current = {
                    "name": m.group(1).strip().strip("\"'"),
                    "splunkbase_id": "",
                    "version": "",
                    "url": "",
                }
                continue

            if current is None:
                continue

            for field in ("splunkbase_id", "version", "url"):
                m = re.match(rf"^\s+{field}:\s*(.+)", line)
                if m:
                    current[field] = m.group(1).strip().strip("\"'")
                    break

    if current and current.get("name"):
        deps.append(current)

    return deps


def docker_exec(container: str, cmd: str, timeout: int = 300) -> tuple:
    """Run a command inside a docker container. Returns (returncode, stdout)."""
    full_cmd = ["docker", "exec", container, "bash", "-lc", cmd]
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, "ERROR: command timed out"
    except Exception as e:
        return 1, f"ERROR: {e}"


# ── Splunkbase auth ──────────────────────────────────────────────────

class SplunkbaseAuth:
    """Handles Splunkbase authentication (token cached per session)."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.token = ""

    def login(self) -> bool:
        if self.token:
            return True
        if not self.username or not self.password:
            print("  ERROR: SPLUNKBASE_USERNAME and SPLUNKBASE_PASSWORD required")
            return False

        print("  Authenticating with Splunkbase...")
        data = urllib.parse.urlencode({
            "username": self.username,
            "password": self.password,
        }).encode()
        req = urllib.request.Request(SPLUNKBASE_LOGIN_URL, data=data, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode()
            m = re.search(r"<id>(.*?)</id>", body)
            if m:
                self.token = m.group(1)
                print("  Splunkbase authenticated")
                return True
            else:
                print(f"  ERROR: No token in response: {body[:200]}")
                return False
        except Exception as e:
            print(f"  ERROR: Splunkbase login failed: {e}")
            return False


# ── Download ─────────────────────────────────────────────────────────

def download_from_splunkbase(auth: SplunkbaseAuth, app_id: str,
                              version: str, output_path: str) -> bool:
    """Download an app tarball from Splunkbase to a local file."""
    sb_url = f"https://splunkbase.splunk.com/app/{app_id}/release/{version}/download"
    print(f"  Downloading from: {sb_url}")

    req = urllib.request.Request(sb_url)
    req.add_header("X-Auth-Token", auth.token)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(output_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Downloaded: {output_path} ({size_mb:.1f} MB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"  ERROR: HTTP {e.code} — {e.reason}")
        return False
    except Exception as e:
        print(f"  ERROR: download failed: {e}")
        return False


# ── Splunk REST API ──────────────────────────────────────────────────

def splunk_api(container: str, password: str, endpoint: str,
               method: str = "GET", data: dict = None) -> dict:
    """Call Splunk REST API inside the container via curl. Returns parsed JSON."""
    url = f"{SPLUNK_API_BASE}{endpoint}"
    cmd_parts = [
        "curl", "-sk", "-X", method, url,
        "-u", f"admin:{password}",
        "-d", "output_mode=json",
    ]
    if data:
        for k, v in data.items():
            cmd_parts.extend(["--data-urlencode", f"{k}={v}"])

    cmd = " ".join(f"'{p}'" for p in cmd_parts)
    rc, output = docker_exec(container, cmd, timeout=300)
    if rc != 0 or not output:
        return {"error": output or "empty response"}
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"error": f"invalid JSON: {output[:300]}"}


def get_installed_version(container: str, password: str, app_name: str) -> str:
    """Check if an app is installed and return its version, or empty string."""
    resp = splunk_api(container, password, f"/services/apps/local/{app_name}")
    try:
        entry = resp.get("entry", [{}])[0]
        content = entry.get("content", {})
        return content.get("version", "")
    except (IndexError, KeyError, AttributeError):
        return ""


def install_from_splunkbase(container: str, password: str,
                            sb_token: str, sb_url: str) -> dict:
    """Tell Splunk to install an app directly from Splunkbase."""
    return splunk_api(container, password, "/services/apps/local", method="POST", data={
        "name": sb_url,
        "update": "true",
        "filename": "true",
        "auth": sb_token,
    })


def install_from_tarball(container: str, password: str,
                         app_name: str) -> tuple:
    """Install a local tarball already in /tmp/apps/ via Splunk CLI."""
    cmd = (
        f"/opt/splunk/bin/splunk install app /tmp/apps/{app_name}.tgz"
        f" -update 1 -auth admin:{password}"
    )
    return docker_exec(container, cmd)


# ── Main: find container ─────────────────────────────────────────────

def find_container(compose_file: str, env_file: str, service: str) -> str:
    """Get the running container ID for a compose service."""
    try:
        result = subprocess.run(
            ["docker", "compose", "--env-file", env_file,
             "-f", compose_file, "ps", "-q", service],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


# ── Main: download-only mode ─────────────────────────────────────────

def run_download_only(args, env: dict) -> int:
    """Download app tarballs without installing (no container needed)."""
    username, password = resolve_credentials(args, env)
    sb_auth = SplunkbaseAuth(username=username, password=password)

    if not sb_auth.login():
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    success = 0
    errors = 0

    # Ad-hoc download: --app-id + --version
    if args.app_id:
        versions = args.version or []
        if not versions:
            print("ERROR: --version required with --app-id")
            return 1
        for version in versions:
            print(f"\nDownloading app {args.app_id} v{version}...")
            output_path = os.path.join(
                args.output_dir, f"app-{args.app_id}-v{version}.tgz"
            )
            if download_from_splunkbase(sb_auth, args.app_id, version, output_path):
                success += 1
            else:
                errors += 1

    # Batch download from deps.yml
    else:
        deps = parse_deps(args.deps_file)
        if not deps:
            print("No dependencies found. Use --app-id or provide a deps file.")
            return 1
        for dep in deps:
            name = dep["name"]
            sb_id = dep["splunkbase_id"]
            version = dep["version"]
            url = dep["url"]

            if url:
                output_path = os.path.join(args.output_dir, f"{name}.tgz")
                print(f"\nDownloading {name} from URL: {url}")
                try:
                    urllib.request.urlretrieve(url, output_path)
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"  Downloaded: {output_path} ({size_mb:.1f} MB)")
                    success += 1
                except Exception as e:
                    print(f"  ERROR: download failed: {e}")
                    errors += 1
            elif sb_id and version:
                output_path = os.path.join(
                    args.output_dir, f"app-{sb_id}-v{version}.tgz"
                )
                print(f"\nDownloading {name} (app {sb_id} v{version})...")
                if download_from_splunkbase(sb_auth, sb_id, version, output_path):
                    success += 1
                else:
                    errors += 1
            else:
                print(f"\nSkipping {name}: no url or splunkbase_id+version")
                errors += 1

    print(f"\nDone: {success} downloaded, {errors} errors")
    return 1 if errors else 0


# ── Main: install mode ───────────────────────────────────────────────

def run_install(args, env: dict) -> int:
    """Download and install deps into a running Splunk container."""
    os.environ.update(env)
    password = env.get("SPLUNK_PASSWORD", os.environ.get("SPLUNK_PASSWORD", ""))

    deps = parse_deps(args.deps_file)
    if not deps:
        print("No dependencies to install.")
        return 0

    container = find_container(args.compose_file, args.env_file, args.service)
    os.makedirs(args.stage_dir, exist_ok=True)

    username, sb_password = resolve_credentials(args, env)
    sb_auth = SplunkbaseAuth(username=username, password=sb_password)

    installed = 0
    skipped = 0
    errors = 0

    for dep in deps:
        name = dep["name"]
        version = dep["version"]
        sb_id = dep["splunkbase_id"]
        url = dep["url"]

        print(f"Checking dependency: {name}")

        current_ver = get_installed_version(container, password, name)
        if current_ver and current_ver == version:
            print(f"  Already installed: {name} v{current_ver} (skipping)")
            skipped += 1
            continue

        if url:
            tar_path = os.path.join(args.stage_dir, f"{name}.tgz")
            print(f"  Downloading from URL: {url}")
            try:
                urllib.request.urlretrieve(url, tar_path)
            except Exception as e:
                print(f"  ERROR: download failed: {e}")
                errors += 1
                continue
            print(f"  Installing {name} ...")
            rc, out = install_from_tarball(container, password, name)
            if rc == 0:
                print(f"  Installed {name}")
                installed += 1
            else:
                print(f"  ERROR: install failed: {out}")
                errors += 1

        elif sb_id and version:
            if not sb_auth.login():
                errors += 1
                continue
            sb_url = f"https://splunkbase.splunk.com/app/{sb_id}/release/{version}/download"
            print(f"  Installing from Splunkbase: {name} v{version}")
            resp = install_from_splunkbase(container, password, sb_auth.token, sb_url)
            if "error" in resp:
                print(f"  ERROR: {resp['error']}")
                errors += 1
            else:
                entry = resp.get("entry", [{}])
                label = entry[0].get("content", {}).get("label", name) if entry else name
                print(f"  Installed {label}")
                installed += 1

        else:
            print(f"  ERROR: need either url or splunkbase_id+version")
            errors += 1

    print(f"\nDependency install complete: {installed} installed, {skipped} skipped, {errors} errors")
    if errors:
        print("If Splunk requests a restart: task dev:restartd")
    return 1 if errors else 0


# ── Entrypoint ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Install or download Splunkbase dependencies"
    )

    # Mode
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download tarballs only (no container/install needed)"
    )

    # Credentials (override .env / env vars / fallback)
    parser.add_argument("--username", help="Splunkbase username")
    parser.add_argument("--password", help="Splunkbase password")

    # Shared
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--deps-file", default="splunk/config/deps.yml")

    # Download-only mode: ad-hoc app
    parser.add_argument("--app-id", help="Splunkbase app ID (download-only)")
    parser.add_argument(
        "--version", action="append",
        help="Version to download (repeatable, download-only)"
    )
    parser.add_argument("--output-dir", default="./downloads",
                        help="Output directory for downloaded tarballs")

    # Install mode
    parser.add_argument("--stage-dir", default="splunk/stage")
    parser.add_argument("--compose-file",
                        default=".devcontainer/docker-compose.yml")
    parser.add_argument("--service", default="splunk")

    args = parser.parse_args()
    env = load_env(args.env_file)

    if args.download_only:
        return run_download_only(args, env)
    else:
        return run_install(args, env)


if __name__ == "__main__":
    sys.exit(main())
