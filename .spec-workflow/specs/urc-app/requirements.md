# Universal REST Client (URC) App — Requirements

## Overview

A Splunk add-on built with UCC SDK that provides a declarative, config-driven way to ingest data from any REST API into Splunk. Uses the Airbyte CDK's declarative framework (vendored) for connector logic, with KV Store checkpointing for incremental collection.

## Goals

- **Zero-code REST API connectors**: Users define API connections via YAML/JSON manifests (Airbyte declarative format) — no Python required per-connector
- **UCC-managed configuration**: Accounts, inputs, proxy, and logging managed through UCC's standard configuration UI
- **Reliable incremental ingestion**: Checkpoint-based state tracking via Splunk KV Store so collection resumes where it left off
- **Template dogfooding**: Built using the splunk-devcontainer-template's Taskfile primitives to validate the developer experience

## Functional Requirements

### FR-1: UCC Configuration Layer
- Account management (REST API credentials, OAuth tokens)
- Input definitions (API endpoint, collection interval, index, sourcetype)
- Proxy and logging configuration via UCC standard pages

### FR-2: Declarative Connector Engine
- Parse Airbyte-format YAML manifests defining streams, auth, pagination, record extraction
- Support auth types: API key, Bearer token, Basic auth, OAuth2
- Support pagination: offset, cursor, page-number, next-link
- Support record extraction via dpath expressions

### FR-3: Checkpointing & State
- KV Store-backed checkpointer (compatible with UCC's KVStoreCheckpointer API)
- Map Airbyte state messages to KV Store entries
- Resume collection from last checkpoint on restart

### FR-4: Data Output
- Write extracted records to Splunk events with configurable index/sourcetype
- Preserve source record structure (JSON events)

## Non-Functional Requirements

### NF-1: Packaging
- Single SPL/tgz distributable via Splunkbase
- Hybrid UCC + vendored Airbyte CDK build pipeline (CIMplicity-style)

### NF-2: Platform Support
- Splunk Enterprise 9.x on Linux (primary)
- Python 3.9+ (Splunk's bundled Python)

### NF-3: Developer Experience
- All build/test/stage operations via `task` commands
- Spec-driven development with progress tracking
