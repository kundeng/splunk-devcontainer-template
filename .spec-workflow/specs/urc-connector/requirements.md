# URC Connector Implementation — Requirements

## Introduction

Implement the core connector functionality for the Universal REST Client (URC) Splunk add-on. This builds on the scaffold (urc-scaffold spec) which established the UCC app structure, vendored the Airbyte CDK declarative runtime, and verified all imports work on Splunk 10.2.

The goal is a working end-to-end pipeline: user configures an API connection via UCC UI, provides an Airbyte declarative manifest, and data flows into a Splunk index with checkpoint-based incremental collection.

## Requirements

### R1: UCC Configuration — Account Entity

**User Story:** As a Splunk admin, I want to configure API credentials (API key, Bearer token, Basic auth, OAuth2) through the UCC UI so that the connector can authenticate with target REST APIs.

#### Acceptance Criteria

1. WHEN a user opens the Configuration page THEN the system SHALL show an Accounts tab with add/edit/delete
2. WHEN a user adds an account THEN the system SHALL present auth_type selection (api_key, bearer, basic, oauth2) with conditional fields
3. WHEN credentials are saved THEN sensitive fields (api_key, bearer_token, password, client_secret) SHALL be encrypted by UCC
4. WHEN an account is referenced by an input THEN the modular input SHALL be able to read decrypted credentials at runtime

### R2: UCC Configuration — Input Entity

**User Story:** As a Splunk admin, I want to define REST API collection inputs with a base URL, Airbyte manifest, collection interval, target index, and sourcetype so that data ingestion is fully configurable.

#### Acceptance Criteria

1. WHEN a user creates a new input THEN the system SHALL present fields for: name, account (dropdown), interval, index, sourcetype, base_url, manifest (textarea)
2. WHEN manifest YAML is provided THEN it SHALL be stored in inputs.conf and readable by the modular input at runtime
3. WHEN interval is set THEN the modular input SHALL execute on that schedule

### R3: Modular Input — Collection Engine

**User Story:** As a Splunk admin, I want the connector to execute Airbyte declarative manifests against REST APIs and write the results as Splunk events so that API data appears in my indexes.

#### Acceptance Criteria

1. WHEN a configured input runs THEN the system SHALL parse the manifest YAML, resolve account credentials, and execute the declarative source
2. WHEN the declarative source yields records THEN each record SHALL be written as a JSON event to the configured index/sourcetype
3. IF the manifest references `{{ config['base_url'] }}` or `{{ config['api_key'] }}` THEN the system SHALL interpolate these from the UCC account and input configuration
4. IF an API request fails (401, 429, 500) THEN the system SHALL follow the manifest's error handling and retry/backoff configuration

### R4: Checkpointing — Incremental Collection

**User Story:** As a Splunk admin, I want the connector to resume collection from where it left off after restarts so that I don't get duplicate or missed data.

#### Acceptance Criteria

1. WHEN a stream has `incremental_sync` configured THEN the system SHALL persist cursor state to Splunk KV Store after each collection cycle
2. WHEN the input restarts THEN the system SHALL load the last checkpoint and resume from the saved cursor position
3. WHEN no checkpoint exists THEN the system SHALL start from the manifest's `start_datetime` or beginning of data

### R5: End-to-End Verification

**User Story:** As a developer, I want to test the full pipeline against a real public API to confirm everything works together.

#### Acceptance Criteria

1. WHEN a manifest for JSONPlaceholder (or similar public API) is configured THEN data SHALL appear in the Splunk index
2. WHEN the input is stopped and restarted with incremental sync THEN no duplicate records SHALL appear
3. WHEN the UCC UI is used to configure and enable the input THEN no CLI intervention SHALL be required

## Non-Functional Requirements

### Performance
- Single input should handle APIs returning up to 10,000 records per collection cycle without timeout
- Checkpoint save should not significantly impact collection throughput

### Reliability
- Input should survive splunkd restarts and resume cleanly
- Transient API errors should not crash the input — retry with backoff

### Developer Experience
- Build and test via existing task primitives (`task ucc:build`, `task ucc:dev`)
- Manifest errors should produce clear log messages with field paths
