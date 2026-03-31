# URC Scaffold — Requirements

## Introduction

Bootstrap the Universal REST Client (URC) Splunk add-on project: scaffold the UCC app, download and vendor the Airbyte CDK declarative runtime and its dependencies, verify everything builds and loads in Splunk 10.2. This is a spike/proof-of-concept — no connector logic yet, just the foundation.

## Requirements

### R1: Scaffold UCC App

**User Story:** As a developer, I want a UCC add-on skeleton (`urc_app`) scaffolded via `ucc-gen init` so that I have a working starting point with globalConfig.json.

#### Acceptance Criteria

1. WHEN `task ucc:init APP_NAME=urc_app` is run THEN the system SHALL create `ucc/urc_app/` with globalConfig.json, package/, app.manifest
2. WHEN `task ucc:build APP_NAME=urc_app` is run THEN the system SHALL produce valid build output in `ucc/output/urc_app/`

### R2: Download Airbyte Declarative Schema

**User Story:** As a developer, I want the Airbyte declarative component schema YAML stored in this repo so that it can be used for code generation and reference.

#### Acceptance Criteria

1. WHEN the schema is downloaded THEN the system SHALL store it at a known path within the repo (e.g., `ucc/urc_app/schema/`)
2. IF the schema file exists THEN it SHALL be the canonical `declarative_component_schema.yaml` from `airbytehq/airbyte-python-cdk`

### R3: Vendor Airbyte CDK Declarative Runtime

**User Story:** As a developer, I want the Airbyte CDK's declarative runtime and its pure-Python dependencies installed into the add-on's `lib/` directory so that I can verify what actually works inside Splunk's Python.

#### Acceptance Criteria

1. WHEN `pip install --target` is run with airbyte-cdk THEN the system SHALL report total size, list of packages with C extensions vs pure Python
2. WHEN C extensions are identified THEN the system SHALL document which are required vs optional for the declarative runtime path
3. WHEN a minimal pure-Python subset is identified THEN it SHALL be captured in `requirements.txt` (or equivalent)
4. IF the vendored deps can be trimmed (exclude unnecessary packages) THEN the trimmed set SHALL be documented in `exclude.txt`

### R4: Build and Load Verification

**User Story:** As a developer, I want to verify the UCC add-on with vendored deps builds successfully and loads in Splunk 10.2 without errors.

#### Acceptance Criteria

1. WHEN `task ucc:build APP_NAME=urc_app` is run with the vendored deps THEN the build SHALL complete without errors
2. WHEN the built add-on is loaded in Splunk 10.2 THEN `import_declare_test` SHALL successfully import and the app SHALL appear in Splunk Web
3. WHEN `import airbyte_cdk.sources.declarative` is attempted from within Splunk's Python THEN the result (success or specific failure) SHALL be documented

### R5: Dependency Audit Report

**User Story:** As a developer, I want a clear report of what works, what doesn't, and what patches are needed so that the full URC design can be informed by real data.

#### Acceptance Criteria

1. WHEN the spike is complete THEN a report SHALL document: total vendored size, pure-Python vs C-extension packages, which imports succeed under Splunk's Python 3.9, and recommended patches
2. IF `pydantic.v1` compatibility works with standalone pydantic v1 THEN this SHALL be confirmed with a test import
3. IF orjson fallback to stdlib json works THEN this SHALL be confirmed

## Non-Functional Requirements

### Developer Experience
- All operations via `task` commands where possible
- Results captured as files in the repo (not just terminal output)

### Scope Boundary
- No connector logic, no auth strategies, no pagination — just scaffolding and dependency verification
- No custom React UI — UCC auto-generated UI only
- No Splunkbase publishing readiness — dev/local only
