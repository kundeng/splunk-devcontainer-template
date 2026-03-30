#!/usr/bin/env python3
"""Test manifest validation and live API collection.

Usage:
    # Validate all manifests parse correctly (no network):
    python test_manifests.py --validate

    # Test JSONPlaceholder (no auth needed):
    python test_manifests.py --live jsonplaceholder

    # Test GitHub (needs GITHUB_TOKEN env var):
    python test_manifests.py --live github

    # Test all that have credentials available:
    python test_manifests.py --live all
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Add URC lib to path
LIB_DIR = Path(__file__).parent.parent / "package" / "lib"
sys.path.insert(0, str(LIB_DIR))

from urc.manifest import process_manifest
from urc.engine import collect

MANIFESTS_DIR = Path(__file__).parent / "manifests"


def validate_manifest(name: str, yaml_path: Path) -> bool:
    """Validate a manifest parses and resolves without errors."""
    print(f"\n{'='*60}")
    print(f"Validating: {name}")
    print(f"{'='*60}")
    try:
        with open(yaml_path) as f:
            manifest_yaml = f.read()
        manifest = process_manifest(manifest_yaml)
        streams = manifest.get("streams", [])
        print(f"  OK: {len(streams)} stream(s) defined")
        for s in streams:
            s_name = s.get("name", "unnamed")
            retriever = s.get("retriever", {})
            paginator = retriever.get("paginator", {})
            pag_type = paginator.get("pagination_strategy", {}).get("type", "none")
            auth = retriever.get("requester", {}).get("authenticator", {}).get("type", "none")
            inc = s.get("incremental_sync", {}).get("type", "none")
            partition = retriever.get("partition_router", {}).get("type", "none")
            print(f"  Stream '{s_name}': auth={auth}, pagination={pag_type}, "
                  f"incremental={inc}, partition={partition}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_live(name: str, yaml_path: Path, config: dict, max_records: int = 50) -> dict:
    """Test a manifest against a live API."""
    print(f"\n{'='*60}")
    print(f"Live test: {name}")
    print(f"Config: {', '.join(f'{k}={v[:20]}...' if len(str(v)) > 20 else f'{k}={v}' for k, v in config.items() if k not in ('api_key', 'password', 'client_secret', 'bearer_token'))}")
    print(f"{'='*60}")

    with open(yaml_path) as f:
        manifest_yaml = f.read()

    results = {"streams": {}, "errors": [], "total_records": 0}
    record_count = 0

    try:
        for stream_name, record, state in collect(manifest_yaml, config):
            if not record:  # Final state emission
                continue
            record_count += 1
            results["total_records"] = record_count

            if stream_name not in results["streams"]:
                results["streams"][stream_name] = {
                    "count": 0,
                    "sample": None,
                    "last_state": None,
                    "fields": set(),
                }
            stream_result = results["streams"][stream_name]
            stream_result["count"] += 1
            stream_result["fields"].update(record.keys())
            if stream_result["sample"] is None:
                stream_result["sample"] = record
            if state:
                stream_result["last_state"] = state

            if record_count >= max_records:
                print(f"  Reached max_records limit ({max_records}), stopping.")
                break

    except Exception as e:
        results["errors"].append(str(e))
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # Print results
    for s_name, s_data in results["streams"].items():
        s_data["fields"] = sorted(s_data["fields"])  # Convert set to list
        print(f"\n  Stream '{s_name}': {s_data['count']} records")
        print(f"  Fields: {', '.join(s_data['fields'][:15])}{'...' if len(s_data['fields']) > 15 else ''}")
        if s_data["sample"]:
            sample_str = json.dumps(s_data["sample"], indent=2, default=str)
            if len(sample_str) > 500:
                sample_str = sample_str[:500] + "..."
            print(f"  Sample record:\n{sample_str}")
        if s_data["last_state"]:
            print(f"  Last state: {json.dumps(s_data['last_state'])}")

    if not results["errors"]:
        print(f"\n  SUCCESS: {record_count} total records collected")
    else:
        print(f"\n  FAILED with {len(results['errors'])} error(s)")

    return results


def get_live_configs():
    """Build config dicts from environment variables."""
    configs = {}

    # JSONPlaceholder — no auth needed
    configs["jsonplaceholder"] = {
        "file": "jsonplaceholder_paginated.yaml",
        "config": {"base_url": "https://jsonplaceholder.typicode.com"},
        "max_records": 100,
    }

    # GitHub — needs GITHUB_TOKEN
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        configs["github"] = {
            "file": "github_repos_issues.yaml",
            "config": {
                "base_url": "https://api.github.com",
                "api_key": token,
                "github_org": os.environ.get("GITHUB_ORG", "anthropics"),
            },
            "max_records": 100,
        }

    # ServiceNow — needs SNOW_* vars
    snow_user = os.environ.get("SNOW_USERNAME", "")
    snow_pass = os.environ.get("SNOW_PASSWORD", "")
    snow_url = os.environ.get("SNOW_URL", "")
    if snow_user and snow_pass and snow_url:
        configs["servicenow"] = {
            "file": "servicenow_incidents.yaml",
            "config": {
                "base_url": snow_url,
                "username": snow_user,
                "password": snow_pass,
            },
            "max_records": 50,
        }

    # Infoblox — needs INFOBLOX_* vars
    ib_user = os.environ.get("INFOBLOX_USERNAME", "")
    ib_pass = os.environ.get("INFOBLOX_PASSWORD", "")
    ib_url = os.environ.get("INFOBLOX_URL", "")
    if ib_user and ib_pass and ib_url:
        configs["infoblox"] = {
            "file": "infoblox_wapi.yaml",
            "config": {
                "base_url": ib_url,
                "username": ib_user,
                "password": ib_pass,
            },
            "max_records": 100,
        }

    # Azure Event Hub — needs AZURE_* vars
    az_client = os.environ.get("AZURE_CLIENT_ID", "")
    az_secret = os.environ.get("AZURE_CLIENT_SECRET", "")
    az_tenant = os.environ.get("AZURE_TENANT_ID", "")
    if az_client and az_secret and az_tenant:
        configs["azure_eventhub"] = {
            "file": "azure_eventhub.yaml",
            "config": {
                "client_id": az_client,
                "client_secret": az_secret,
                "tenant_id": az_tenant,
                "subscription_id": os.environ.get("AZURE_SUBSCRIPTION_ID", ""),
                "resource_group": os.environ.get("AZURE_RESOURCE_GROUP", ""),
                "eventhub_namespace": os.environ.get("AZURE_EVENTHUB_NAMESPACE", ""),
                "eventhub_name": os.environ.get("AZURE_EVENTHUB_NAME", ""),
            },
            "max_records": 50,
        }

    return configs


def main():
    parser = argparse.ArgumentParser(description="Test URC manifests")
    parser.add_argument("--validate", action="store_true", help="Validate all manifests parse correctly")
    parser.add_argument("--live", type=str, help="Run live API test (jsonplaceholder, github, servicenow, infoblox, azure_eventhub, all)")
    args = parser.parse_args()

    if not args.validate and not args.live:
        args.validate = True

    # Collect all manifests
    manifest_files = sorted(MANIFESTS_DIR.glob("*.yaml"))
    if not manifest_files:
        print("No manifest files found in", MANIFESTS_DIR)
        return 1

    exit_code = 0

    if args.validate:
        print("Validating all manifests...")
        for f in manifest_files:
            if not validate_manifest(f.stem, f):
                exit_code = 1

    if args.live:
        configs = get_live_configs()
        targets = [args.live] if args.live != "all" else list(configs.keys())

        for target in targets:
            if target not in configs:
                print(f"\nSkipping '{target}' — credentials not configured")
                print(f"  Set environment variables to enable this test")
                continue
            tc = configs[target]
            results = test_live(
                target,
                MANIFESTS_DIR / tc["file"],
                tc["config"],
                tc.get("max_records", 50),
            )
            if results["errors"]:
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
