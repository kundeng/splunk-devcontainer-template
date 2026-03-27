#!/usr/bin/env python3
"""Download apps from Splunkbase to local filesystem.

Usage:
    python3 splunkbase-download.py --app-id 7404 --version 3.4.1 [--env-file .env] [--output-dir ./downloads]

Credentials resolved in order:
  1. --username / --password CLI args
  2. SPLUNKBASE_USERNAME / SPLUNKBASE_PASSWORD in .env file
  3. SPLUNKBASE_USERNAME / SPLUNKBASE_PASSWORD environment variables
  4. Built-in fallback account

No external dependencies — stdlib only (Python 3.8+).

Based on deps-install.py from splunk-devcontainer-template.
"""

import argparse
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


# ── Config ───────────────────────────────────────────────────────────

SPLUNKBASE_LOGIN_URL = "https://splunkbase.splunk.com/api/account:login/"
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
    """Download an app from Splunkbase to a local file."""
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


# ── Main ─────────────────────────────────────────────────────────────

def resolve_credentials(args, env: dict) -> tuple:
    """Resolve Splunkbase credentials from CLI args, .env, environ, or fallback."""
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


def main():
    parser = argparse.ArgumentParser(description="Download apps from Splunkbase")
    parser.add_argument("--app-id", required=True, help="Splunkbase app ID (e.g. 7404)")
    parser.add_argument("--version", required=True, action="append",
                        help="Version to download (can be specified multiple times)")
    parser.add_argument("--username", help="Splunkbase username (overrides .env)")
    parser.add_argument("--password", help="Splunkbase password (overrides .env)")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="./downloads")
    args = parser.parse_args()

    # Load .env (optional — no error if missing)
    env = load_env(args.env_file)

    username, password = resolve_credentials(args, env)
    sb_auth = SplunkbaseAuth(username=username, password=password)

    if not sb_auth.login():
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    success = 0
    errors = 0

    for version in args.version:
        print(f"\nDownloading app {args.app_id} v{version}...")
        output_path = os.path.join(args.output_dir, f"app-{args.app_id}-v{version}.tgz")
        if download_from_splunkbase(sb_auth, args.app_id, version, output_path):
            success += 1
        else:
            errors += 1

    print(f"\nDone: {success} downloaded, {errors} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
