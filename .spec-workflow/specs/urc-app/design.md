# Universal REST Client (URC) App — Design

## Overview

The URC app is a Splunk add-on that lets users ingest data from any REST API using declarative YAML manifests (Airbyte-compatible format). It combines UCC SDK for configuration management with a custom connector engine backed by a typed schema layer generated from Airbyte's canonical component schema (109 types). KV Store provides checkpoint state for incremental collection.

Target platform: **Splunk Enterprise 10.2+** (Python 3.9 LTS default, 3.13 opt-in).

The app builds on the existing `splunk-devcontainer-template` infrastructure — UCC tasks, Docker Compose, symlink-based dev, and Taskfile primitives.

## Code Reuse Analysis

### Existing Components to Leverage

- **`task ucc:*` primitives**: `ucc:init`, `ucc:build`, `ucc:link`, `ucc:dev`, `ucc:package` — already working
- **`task app:sync-links`**: Handles symlink + ACL setup for bind-mounted apps
- **`task stage:deploy`**: Full packaging + staging pipeline
- **Docker Compose mounts**: `ucc/output:/opt/splunk/ucc-apps` already wired
- **splunktaucclib**: BaseModInput, AdminExternalHandler — installed by ucc-gen build
- **solnlib**: KVStoreCheckpointer, ConfManager, CredentialManager — installed by ucc-gen build

### Integration Points

- **Splunk REST API**: UCC-generated REST handlers for CRUD on accounts, inputs, settings
- **KV Store REST API**: Checkpoint persistence via solnlib.KVStoreCheckpointer
- **Splunk HEC / event writer**: Output records as JSON events to configured index/sourcetype
- **UCC modular input framework**: BaseModInput drives the collection loop

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Splunk Web                         │
│  UCC Auto-Generated UI (accounts, inputs, settings)  │
└──────────────────┬───────────────────────────────────┘
                   │ REST (splunkd)
┌──────────────────▼───────────────────────────────────┐
│              UCC REST Handlers                        │
│  (auto-generated from globalConfig.json)              │
│  CRUD on: accounts.conf, inputs.conf, settings.conf   │
└──────────────────┬───────────────────────────────────┘
                   │ reads config
┌──────────────────▼───────────────────────────────────┐
│           Modular Input Handler                       │
│  urc_input.py (BaseModInput)                          │
│                                                       │
│  1. Read account + input config from .conf            │
│  2. Load + validate YAML manifest (Pydantic v1 models)│
│  3. Build connector engine (auth, paginator, etc.)    │
│  4. Execute collection loop:                          │
│     ├─ Load checkpoint (KV Store)                     │
│     ├─ Fetch pages via connector engine               │
│     ├─ Extract records via dpath                      │
│     ├─ Write events to Splunk (event_writer)          │
│     └─ Save checkpoint (KV Store)                     │
└──────────────────┬───────────────────────────────────┘
                   │ uses
┌──────────────────▼───────────────────────────────────┐
│          Vendored Libraries (package/lib/)             │
│                                                       │
│  connector/           ← Our connector engine          │
│  ├── engine.py        ← Manifest→execution pipeline   │
│  ├── auth.py          ← Auth strategy factory         │
│  ├── pagination.py    ← Pagination strategy factory   │
│  ├── extraction.py    ← dpath record extraction       │
│  └── checkpoint.py    ← KV Store checkpoint bridge    │
│                                                       │
│  schema/              ← Typed manifest models         │
│  ├── models.py        ← Generated from Airbyte schema │
│  ├── extensions.py    ← Our custom types beyond AB    │
│  └── validate.py      ← Load + validate + merge      │
│                                                       │
│  pydantic/ (v1)       ← Pure Python, 166 KB           │
│  typing_extensions/   ← Pure Python                   │
│  dpath/               ← Pure Python, MIT, zero deps   │
│  pyyaml, requests     ← Via requirements.txt          │
└──────────────────────────────────────────────────────┘
```

### Decision: Typed Schema Layer via Pydantic v1

We download Airbyte's canonical `declarative_component_schema.yaml` (JSON Schema Draft 7, 109 component types) and generate typed Python models using `datamodel-codegen` at dev time.

**Runtime validation library: Pydantic v1** (1.10.x)
- **Pure Python** `py3-none-any` wheel — 166 KB, zero C extensions, zero `.so` files
- Only runtime dep: `typing-extensions` (also pure Python)
- Built-in discriminated unions via `Field(discriminator="type")`
- Clear error messages with field paths
- Works on Python 3.7–3.13 (covers all Splunk versions)
- Same library Airbyte uses internally for their generated models

**Dev-time code generation pipeline:**
1. Download `declarative_component_schema.yaml` from `airbytehq/airbyte-python-cdk`
2. Run `datamodel-codegen --input schema.yaml --output models.py --target-python-version 3.9 --use-annotated --output-model-type pydantic.BaseModel`
3. Generated models go into `package/lib/schema/models.py`
4. Custom extensions (our own component types) in `schema/extensions.py`
5. Regenerate when upstream schema updates (Taskfile target: `urc:update-schema`)

**What this gives us:**
- Full type checking of manifests at load time (not runtime surprises)
- IDE autocomplete when editing connector code
- Extensibility — add custom auth types, paginators, extractors beyond Airbyte's 109
- Manifest validation with clear error messages ("stream[0].retriever.paginator.page_size: field required")
- Forward compatibility — regenerate models when Airbyte adds new types

### Decision: UCC Auto-Generated UI (Phase 1)

Use the **UCC auto-generated UI** for accounts, inputs, and settings. Manifest YAML goes in a textarea field on the input form. A custom React wizard for manifest editing/import can be added as a Phase 2 enhancement.

**Rationale:** Ship a working connector faster. The UCC UI handles all CRUD operations out of the box.

### Decision: Custom Connector Engine (not vendored Airbyte CDK)

Build a slim connector engine that reads Airbyte-format manifests but implements execution logic natively using `requests`, `pyyaml`, `dpath`, and our generated Pydantic models.

**Rationale:**
- Airbyte CDK requires Python 3.10+ and pulls in pydantic v2 (Rust), Jinja2, jsonschema, etc.
- We only need: HTTP requests with auth, pagination, dpath extraction, cursor checkpointing
- The typed schema layer gives us Airbyte manifest compatibility without the CDK runtime
- 95% of declarative manifests use a small subset of features we implement directly
- Full CDK compatibility can be added later if needed

**What we keep from Airbyte:**
- Exact YAML manifest format (validated against their schema)
- All 109 component type definitions (as Pydantic models)
- dpath field_path syntax for record extraction
- Cursor-based checkpoint model (DatetimeBasedCursor, IncrementingCountCursor)

## Components and Interfaces

### Component 1: Schema Layer (`package/lib/schema/`)

- **Purpose:** Typed manifest models generated from Airbyte's schema + our extensions
- **Location:** `package/lib/schema/`

| File | Purpose |
|------|---------|
| `models.py` | Generated Pydantic v1 models (109 Airbyte types) |
| `extensions.py` | Custom types beyond Airbyte (e.g., SplunkHecOutput) |
| `validate.py` | Load YAML → parse → validate → return typed model |

- **Interfaces:**
  ```python
  # schema/validate.py
  def load_manifest(yaml_string: str) -> DeclarativeSource:
      """Parse YAML, validate against schema, return typed model.
      Raises ValidationError with field paths on invalid input."""

  def merge_config(manifest: DeclarativeSource, config: dict) -> DeclarativeSource:
      """Resolve {{ config['key'] }} interpolations in manifest fields."""
  ```

### Component 2: globalConfig.json

- **Purpose:** Define all UCC-managed configuration — accounts, inputs, settings
- **Location:** `ucc/urc_app/globalConfig.json`

**Account entity** (multiple auth types via `options`):

| Field | Type | Description |
|-------|------|-------------|
| name | text | Account name (unique key) |
| auth_type | singleSelect | api_key, bearer, basic, oauth2 |
| api_key | text (encrypted) | For auth_type=api_key |
| bearer_token | text (encrypted) | For auth_type=bearer |
| username | text | For auth_type=basic |
| password | text (encrypted) | For auth_type=basic |
| client_id | text | For auth_type=oauth2 |
| client_secret | text (encrypted) | For auth_type=oauth2 |
| token_url | text | For auth_type=oauth2 |
| custom_headers | text | Optional JSON key-value pairs |

**Input entity:**

| Field | Type | Description |
|-------|------|-------------|
| name | text | Input name (unique key) |
| account | singleSelect | Reference to account entity |
| interval | interval | Collection interval (seconds) |
| index | singleSelect | Target Splunk index |
| sourcetype | text | Event sourcetype |
| base_url | text | API base URL |
| manifest | textarea | YAML manifest content (Airbyte format) |

**Settings tabs:** Logging (log level), Proxy (host, port, user, pass, type, rdns)

### Component 3: Connector Engine (`package/lib/connector/engine.py`)

- **Purpose:** Take a validated manifest model → build configured HTTP pipeline → execute collection
- **Interfaces:**
  ```python
  class ConnectorEngine:
      def __init__(self, source: DeclarativeSource, auth_config: dict, logger)
      def collect(self, checkpoint: dict | None) -> Iterator[tuple[dict, dict]]
          # yields (record, updated_state) tuples
  ```
- **Dependencies:** `connector.auth`, `connector.pagination`, `connector.extraction`, `schema.models`

### Component 4: Auth Strategies (`package/lib/connector/auth.py`)

- **Purpose:** Factory for request authentication based on manifest authenticator model
- **Interfaces:**
  ```python
  class AuthFactory:
      @staticmethod
      def from_model(authenticator: AuthenticatorModel, config: dict) -> AuthStrategy

  class AuthStrategy(Protocol):
      def apply(self, session: requests.Session, request_kwargs: dict) -> dict
  ```
- **Supported types:** ApiKeyAuth, BearerAuth, BasicAuth, OAuth2Auth (maps 1:1 to Airbyte schema types)

### Component 5: Pagination Strategies (`package/lib/connector/pagination.py`)

- **Purpose:** Factory for pagination based on manifest paginator model
- **Interfaces:**
  ```python
  class PaginatorFactory:
      @staticmethod
      def from_model(paginator: PaginatorModel) -> Paginator

  class Paginator(Protocol):
      def next_page(self, response: requests.Response, current_params: dict) -> dict | None
          # returns updated params for next page, or None if done
  ```
- **Supported types:** OffsetPaginator, CursorPaginator, PageNumberPaginator, NextLinkPaginator

### Component 6: Record Extraction (`package/lib/connector/extraction.py`)

- **Purpose:** Extract records from API responses using dpath field paths
- **Interfaces:**
  ```python
  def extract_records(response_body: dict, field_path: list[str]) -> list[dict]
  ```
- **Dependencies:** `dpath` (vendored, MIT, zero deps, 15 KB)

### Component 7: Checkpoint Bridge (`package/lib/connector/checkpoint.py`)

- **Purpose:** Bridge between connector engine cursor state and Splunk KV Store
- **Interfaces:**
  ```python
  class CheckpointManager:
      def __init__(self, checkpointer: KVStoreCheckpointer, stream_name: str)
      def load(self) -> dict | None
      def save(self, state: dict) -> None
  ```
- **Dependencies:** `solnlib.KVStoreCheckpointer`

### Component 8: Modular Input Handler (`package/bin/urc_input.py`)

- **Purpose:** UCC modular input entry point — ties all components together
- **Interfaces:** Implements `BaseModInput.collect_events()` or stream-based input
- **Flow:**
  1. Import `import_declare_test` (required UCC pattern)
  2. Read input config (account, base_url, manifest, interval)
  3. Read account credentials via `ConfManager`
  4. Parse + validate manifest YAML via `schema.validate.load_manifest()`
  5. Resolve config interpolations via `schema.validate.merge_config()`
  6. Create `ConnectorEngine` with validated source model + auth config
  7. Load checkpoint via `CheckpointManager`
  8. Iterate `engine.collect(checkpoint)` → write events → save state

## Data Models

### Manifest (Airbyte declarative format — validated by generated Pydantic models)

```yaml
version: "0.1.0"
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: "stream_name"
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url_base: "{{ config['base_url'] }}"
        path: "/api/resource"
        http_method: GET
        authenticator:
          type: ApiKeyAuthenticator
          header: "X-API-Key"
          api_token: "{{ config['api_key'] }}"
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: ["data", "items"]
      paginator:
        type: DefaultPaginator
        pagination_strategy:
          type: OffsetIncrement
          page_size: 100
    incremental_sync:
      type: DatetimeBasedCursor
      cursor_field: "updated_at"
      start_datetime: "2024-01-01T00:00:00Z"
```

### Checkpoint (KV Store document)

```json
{
  "_key": "input_name::stream_name",
  "state": {
    "cursor_field": "updated_at",
    "cursor_value": "2024-03-15T10:30:00Z"
  },
  "updated_at": "2024-03-15T10:31:00Z"
}
```

### Account Config (.conf)

```ini
[my_api_account]
auth_type = api_key
api_key = ****  # encrypted by UCC
custom_headers = {"X-Custom": "value"}
```

## Vendored Dependencies

All Python deps are vendored into `package/lib/` (installed by `ucc-gen build` from `requirements.txt`). **No C extensions — pure Python only.**

| Library | Size | License | Purpose |
|---------|------|---------|---------|
| pydantic (v1) | 166 KB | MIT | Manifest validation, typed models, discriminated unions |
| typing-extensions | ~80 KB | PSF | Pydantic v1 runtime dependency |
| dpath | 15 KB | MIT | Record extraction (Airbyte DpathExtractor compatible) |
| requests | ~450 KB | Apache 2 | HTTP client (likely already in Splunk, but vendor for safety) |
| PyYAML | ~600 KB | MIT | Manifest parsing |
| backoff | ~30 KB | MIT | Retry/backoff for API rate limiting |

**Total vendored footprint: ~1.3 MB** (all pure Python)

## Error Handling

### Error Scenarios

1. **Invalid manifest YAML**
   - **Handling:** Pydantic ValidationError with field paths (e.g., `streams[0].retriever.paginator: field required`)
   - **User Impact:** Input fails to start; clear error in logs pointing to exact manifest problem

2. **API authentication failure (401/403)**
   - **Handling:** Log error with account name, skip collection cycle, retry on next interval
   - **User Impact:** Events stop arriving; error visible in internal logs

3. **API rate limiting (429)**
   - **Handling:** Respect Retry-After header; exponential backoff via `backoff` library
   - **User Impact:** Collection slows but continues

4. **Network timeout / connection error**
   - **Handling:** Retry with backoff (3 attempts); log warning; resume from checkpoint on next interval
   - **User Impact:** Temporary gap in data; auto-recovers

5. **KV Store unavailable**
   - **Handling:** Log error; continue collection without checkpoint (may cause duplicates)
   - **User Impact:** Possible duplicate events until KV Store recovers

## Testing Strategy

### Unit Testing

- `test_schema.py` — Validate sample manifests against generated models; test error messages
- `test_auth.py` — Verify each auth strategy applies correct headers/params
- `test_pagination.py` — Verify each paginator produces correct next-page params
- `test_extraction.py` — Verify dpath extraction from sample responses
- `test_engine.py` — Verify engine wires auth + pagination + extraction correctly (mocked HTTP)
- `test_checkpoint.py` — Verify checkpoint save/load round-trip

### Integration Testing

- `test_input_handler.py` — Verify modular input reads config and invokes engine
- Test against mock HTTP server (Python `http.server` or `responses` library)

### End-to-End Testing

- Configure input via Splunk Web (UCC UI) → verify data appears in index
- Test checkpoint resume: stop input, restart, verify no duplicate/missed events
- Test with real public APIs (e.g., JSONPlaceholder, GitHub API)
- Package and install on staging Splunk (`task stage:deploy`)

## File Structure

```
ucc/urc_app/
├── globalConfig.json              # UCC configuration (accounts, inputs, settings)
├── package/
│   ├── app.manifest               # Splunk app metadata
│   ├── bin/
│   │   └── urc_input.py           # Modular input entry point
│   ├── lib/
│   │   ├── requirements.txt       # Python deps (pydantic<2, dpath, requests, pyyaml, backoff)
│   │   ├── schema/
│   │   │   ├── __init__.py
│   │   │   ├── models.py          # Generated Pydantic v1 models (from Airbyte schema)
│   │   │   ├── extensions.py      # Custom types beyond Airbyte spec
│   │   │   └── validate.py        # Load + validate + interpolate manifests
│   │   └── connector/
│   │       ├── __init__.py
│   │       ├── engine.py           # Manifest→execution pipeline
│   │       ├── auth.py             # Auth strategy factory
│   │       ├── pagination.py       # Pagination strategy factory
│   │       ├── extraction.py       # dpath record extraction
│   │       └── checkpoint.py       # KV Store checkpoint bridge
│   ├── default/
│   │   └── restmap.conf            # (if custom REST endpoints needed)
│   └── static/
│       ├── appIcon.png
│       └── appIcon_2x.png
├── schema/                         # Dev-time schema management (not shipped)
│   ├── declarative_component_schema.yaml  # Downloaded from Airbyte CDK
│   └── generate.sh                 # datamodel-codegen → package/lib/schema/models.py
└── README.md
```
