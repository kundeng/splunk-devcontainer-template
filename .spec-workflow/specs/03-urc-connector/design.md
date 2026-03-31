# URC Connector Implementation — Design

## Overview

Wire the Airbyte CDK declarative runtime into the UCC modular input framework. Users configure accounts and inputs via UCC UI, provide an Airbyte manifest YAML, and data flows into Splunk indexes with KV Store checkpointing.

## Code Reuse Analysis

### Existing Components to Leverage

- **`urc_app_input_helper.py`** — existing UCC modular input scaffold with credential access pattern (`conf_manager.ConfManager`, `smi.EventWriter`)
- **`import_declare_test`** — auto-generated sys.path setup, adds lib/ to path
- **Vendored `airbyte_cdk`** — declarative runtime at `package/lib/airbyte_cdk/`
- **`solnlib.KVStoreCheckpointer`** — checkpoint persistence (installed by ucc-gen build)
- **`globalConfig.json`** — existing account + input entity definitions (need updating)

### Integration Points

- **UCC REST handlers** — auto-generated CRUD for accounts.conf and inputs.conf
- **Splunk KV Store** — checkpoint persistence via REST API (solnlib wrapper)
- **`smi.EventWriter`** — write JSON events to Splunk indexes
- **`conf_manager.ConfManager`** — read decrypted credentials at runtime

## Architecture

```
globalConfig.json
  ├─ Account: auth_type, api_key/bearer_token/username+password/oauth2 fields
  └─ Input: name, account, interval, index, sourcetype, base_url, manifest (textarea)
        │
        ▼  (UCC stores in inputs.conf / accounts.conf)
urc_app_input_helper.py :: stream_events()
  │
  ├─ Read input config (account, base_url, manifest, index, sourcetype)
  ├─ Read account credentials via ConfManager (decrypted)
  ├─ Build config dict: merge account creds + input fields
  ├─ Parse manifest YAML
  ├─ Load checkpoint from KV Store
  │
  ├─ Create ConcurrentDeclarativeSource(source_config=manifest, config=config_dict)
  │     └─ CDK handles: auth, pagination, record extraction, error handling, retry
  │
  ├─ source.read() → yields AirbyteMessage objects
  │     ├─ RECORD → write as JSON event via EventWriter
  │     └─ STATE  → save to KV Store checkpoint
  │
  └─ Done. Next interval repeats.
```

## Components and Interfaces

### Component 1: globalConfig.json (updated)

Extend the scaffold's globalConfig with auth type selection and manifest textarea.

**Account entity changes:**

| Field | Type | Description |
|-------|------|-------------|
| name | text | Account name (existing) |
| auth_type | singleSelect | api_key, bearer, basic, oauth2 |
| api_key | text (encrypted) | Shown when auth_type=api_key |
| bearer_token | text (encrypted) | Shown when auth_type=bearer |
| username | text | Shown when auth_type=basic |
| password | text (encrypted) | Shown when auth_type=basic |
| client_id | text | Shown when auth_type=oauth2 |
| client_secret | text (encrypted) | Shown when auth_type=oauth2 |
| token_url | text | Shown when auth_type=oauth2 |

UCC supports `options.dependencies` to show/hide fields based on `auth_type` value.

**Input entity changes (add to existing):**

| Field | Type | Description |
|-------|------|-------------|
| base_url | text | API base URL (e.g., `https://api.example.com/v1`) |
| sourcetype | text | Event sourcetype (default: `urc:api:json`) |
| manifest | textarea | Airbyte declarative manifest YAML |

### Component 2: urc_app_input_helper.py (rewritten)

The main collection logic. Follows the existing UCC pattern but replaces dummy data with CDK execution.

```python
def stream_events(inputs, event_writer):
    for input_name, input_item in inputs.inputs.items():
        session_key = inputs.metadata["session_key"]
        # 1. Read config
        account_config = get_account_config(session_key, input_item["account"])
        config_dict = build_config_dict(account_config, input_item)
        # 2. Parse manifest
        manifest = yaml.safe_load(input_item["manifest"])
        # 3. Load checkpoint
        ckpt = CheckpointManager(session_key, input_name)
        state = ckpt.load()
        # 4. Run declarative source
        source = ConcurrentDeclarativeSource(
            source_config=manifest, config=config_dict
        )
        catalog = build_catalog(manifest)
        for message in source.read(logger, config_dict, catalog, state):
            if message.type == Type.RECORD:
                event_writer.write_event(smi.Event(...))
            elif message.type == Type.STATE:
                ckpt.save(message.state)
```

### Component 3: CheckpointManager

Bridges Airbyte state messages to Splunk KV Store.

```python
class CheckpointManager:
    def __init__(self, session_key, input_name):
        self._ckpt = KVStoreCheckpointer(
            collection_name="urc_checkpoints",
            session_key=session_key,
            app=ADDON_NAME,
        )
        self._key = input_name

    def load(self):
        """Load last state as list of AirbyteStateMessage dicts."""
        data = self._ckpt.get(self._key)
        return data.get("state", []) if data else []

    def save(self, state_message):
        """Save AirbyteStateMessage to KV Store."""
        self._ckpt.update(self._key, {"state": [state_message.dict()]})
```

### Component 4: Config Builder

Maps UCC account config to the `config` dict that CDK manifest interpolation expects (`{{ config['api_key'] }}`).

```python
def build_config_dict(account_config, input_item):
    """Merge account credentials + input fields into CDK config dict."""
    return {
        "base_url": input_item.get("base_url", ""),
        "auth_type": account_config.get("auth_type", ""),
        "api_key": account_config.get("api_key", ""),
        "bearer_token": account_config.get("bearer_token", ""),
        "username": account_config.get("username", ""),
        "password": account_config.get("password", ""),
        "client_id": account_config.get("client_id", ""),
        "client_secret": account_config.get("client_secret", ""),
        "token_url": account_config.get("token_url", ""),
    }
```

### Component 5: Catalog Builder

Builds an `AirbyteCatalog` from the manifest's stream definitions so the CDK knows which streams to read.

```python
def build_catalog(manifest):
    """Build ConfiguredAirbyteCatalog from manifest streams."""
    streams = []
    for stream_def in manifest.get("streams", []):
        streams.append(ConfiguredAirbyteStream(
            stream=AirbyteStream(name=stream_def["name"], json_schema={}),
            sync_mode=SyncMode.incremental if "incremental_sync" in stream_def
                      else SyncMode.full_refresh,
            destination_sync_mode=DestinationSyncMode.append,
        ))
    return ConfiguredAirbyteCatalog(streams=streams)
```

## Data Flow

```
UCC UI → accounts.conf + inputs.conf
  → stream_events() reads config
    → ConcurrentDeclarativeSource(manifest, config)
      → CDK resolves {{ config['api_key'] }} etc.
      → CDK makes HTTP requests (auth, pagination, retry)
      → CDK yields AirbyteMessage(RECORD) + AirbyteMessage(STATE)
    → RECORD → smi.Event(data=json, index=..., sourcetype=...)
    → STATE  → KVStoreCheckpointer.update()
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| Invalid manifest YAML | Log parse error, skip input, retry next interval |
| API auth failure (401/403) | CDK raises, caught in try/except, logged, retry next interval |
| API rate limit (429) | CDK handles via manifest backoff config |
| Network error | CDK retries per manifest error_handler config |
| KV Store unavailable | Log warning, continue without checkpoint (may duplicate) |
| CDK import failure | Log error at startup, input disabled |

## Testing Strategy

### Unit Testing
- `test_config_builder.py` — verify config dict merge from account + input
- `test_checkpoint.py` — verify KV Store save/load round-trip (mocked)
- `test_catalog_builder.py` — verify catalog construction from manifest

### Integration Testing
- Configure input with JSONPlaceholder manifest → verify events in index
- Stop/restart input → verify checkpoint resume (no duplicates)
- Test each auth type (at minimum api_key + bearer via mock API)

### E2E Testing
- Full UCC UI flow: create account → create input → enable → verify data
- `task ucc:dev` → verify app loads and collects data

## File Changes

```
ucc/urc_app/
├── globalConfig.json              # MODIFY: add auth_type, manifest, base_url, sourcetype fields
├── package/
│   ├── bin/
│   │   └── urc_app_input_helper.py  # REWRITE: CDK integration
│   ├── lib/
│   │   ├── requirements.txt       # No changes needed
│   │   └── airbyte_cdk/           # Already vendored (subtree)
│   └── default/
│       └── collections.conf       # NEW: define urc_checkpoints KV Store collection
```
