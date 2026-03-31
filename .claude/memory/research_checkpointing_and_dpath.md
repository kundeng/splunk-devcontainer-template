# Research: Checkpointing Patterns & dpath Library

**Date**: 2026-03-27
**Purpose**: Compare Splunk KV Store checkpointing with Airbyte state management; evaluate dpath for record extraction

---

## 1. Splunk KV Store Checkpointing

### 1.1 solnlib.KVStoreCheckpointer API

The current `solnlib` (v4+) API uses `get`/`update`/`delete` (NOT the older `get_state`/`save_state` from deprecated `splunktalib`).

```python
from solnlib.modular_input import checkpointer

# Initialize with collection name, session key, app name
ckpt = checkpointer.KVStoreCheckpointer(
    "my_addon_checkpoints",   # KV store collection name
    session_key,              # From inputs.metadata["session_key"]
    "my_addon",               # App name
    # Optional: scheme="https", host="localhost", port=8089
)

# Save checkpoint state (key=string, state=JSON-serializable dict)
ckpt.update("input_name_1", {"last_timestamp": "2026-03-27T10:00:00Z"})

# Retrieve checkpoint (returns dict or None)
state = ckpt.get("input_name_1")
# => {"last_timestamp": "2026-03-27T10:00:00Z"}

# Delete checkpoint
ckpt.delete("input_name_1")

# Batch update (list of dicts with _key and state)
ckpt.batch_update([
    {"_key": "input_1", "state": {"cursor": "abc123"}},
    {"_key": "input_2", "state": {"cursor": "def456"}},
])
```

**Abstract base**: `Checkpointer(ABCMeta)` defines `update`, `get`, `delete`, `batch_update`.

### 1.2 Underlying Storage: Splunk KV Store

KV Store is a document store (MongoDB-backed internally) accessible via REST API:

- **Collections** = tables. Defined in `collections.conf` or via REST.
- **Records** = rows. Each has a `_key` (unique ID) and arbitrary JSON fields.
- **REST endpoints**:
  - `GET /servicesNS/nobody/<app>/storage/collections/data/<collection>/<key>` -- read
  - `POST /servicesNS/nobody/<app>/storage/collections/data/<collection>` -- create/update
  - `DELETE /servicesNS/nobody/<app>/storage/collections/data/<collection>/<key>` -- delete
- KV Store runs on port 8191, resides on search heads.
- In search head clusters: writes go to KV store captain, reads are local.
- `KVStoreCheckpointer` wraps these REST calls via `splunklib`.

**What KVStoreCheckpointer stores per record:**
```json
{
    "_key": "input_name_1",
    "state": "{\"last_timestamp\": \"2026-03-27T10:00:00Z\"}"
}
```
The `state` field is a JSON-encoded string of the state dict.

### 1.3 splunktaucclib BaseModInput Checkpoint Methods

`BaseModInput` (from `modinput_wrapper/base_modinput.py`) provides convenience wrappers:

```python
class BaseModInput(smi.Script):
    def _init_ckpt(self):
        if os.environ.get("AOB_TEST"):
            self.ckpt = FileCheckpointer(self.checkpoint_dir)
        else:
            self.ckpt = KVStoreCheckpointer(
                self.app + "_checkpointer",  # collection name
                self.session_key,
                self.app
            )

    def get_check_point(self, key):
        return self.ckpt.get(key)        # delegates to solnlib

    def save_check_point(self, key, state):
        return self.ckpt.update(key, state)  # delegates to solnlib

    def delete_check_point(self, key):
        return self.ckpt.delete(key)
```

In UCC-generated add-ons using `inputHelperModule` pattern (like our template), the `helper` object IS the BaseModInput instance. Usage:

```python
def stream_events(helper, event_writer):
    # Retrieve last checkpoint
    state = helper.get_check_point("my_input://prod_endpoint")
    last_ts = state.get("last_timestamp") if state else None

    # Collect data from API (using last_ts as filter)
    records = fetch_from_api(since=last_ts)

    # Write events
    for record in records:
        event_writer.write_event(smi.Event(data=json.dumps(record)))

    # Save new checkpoint
    if records:
        new_ts = max(r["updated_at"] for r in records)
        helper.save_check_point("my_input://prod_endpoint", {
            "last_timestamp": new_ts,
            "record_count": len(records),
        })
```

### 1.4 Typical Checkpoint Data

| Pattern | Checkpoint State | Use Case |
|---------|-----------------|----------|
| Timestamp cursor | `{"last_timestamp": "2026-03-27T10:00:00Z"}` | APIs with `updated_since` param |
| ID cursor | `{"last_id": 54321}` | Auto-incrementing ID pagination |
| Page token | `{"next_page_token": "eyJwYWdl..."}` | APIs with opaque cursor tokens |
| Offset | `{"offset": 500}` | Offset-based pagination across intervals |
| Composite | `{"last_ts": "...", "last_id": 123, "page": 5}` | Complex multi-field cursors |

### 1.5 Alternative Checkpoint Mechanisms

| Method | Storage | Pros | Cons |
|--------|---------|------|------|
| **KVStoreCheckpointer** (recommended) | KV Store (MongoDB) | Persistent, replicated in SHC, REST-accessible | Requires search head, not on UFs |
| **FileCheckpointer** | `$SPLUNK_HOME/var/lib/splunk/modinputs/<input>/` | Works on UFs, simple | Not replicated, lost on reinstall |
| **conf files** | `local/*.conf` via `conf_manager` | Visible in UI | Slow writes, reload needed, not designed for frequent updates |
| **Splunk native checkpoint dir** | `inputs.metadata["checkpoint_dir"]` | Built-in, always available | File-based, same limitations as FileCheckpointer |

**Best practice**: Use `KVStoreCheckpointer` for production add-ons. Use `FileCheckpointer` only for testing (`AOB_TEST=1`) or universal forwarder deployments.

---

## 2. Airbyte State Management

### 2.1 State Message Format

Airbyte uses `AirbyteStateMessage` to persist cursor position between syncs:

```json
{
    "type": "STATE",
    "state": {
        "type": "STREAM",
        "stream": {
            "stream_descriptor": {
                "name": "users",
                "namespace": "public"
            },
            "stream_state": {
                "updated_at": "2026-03-27T10:00:00Z"
            }
        }
    }
}
```

**State types:**
- `STREAM` (per-stream) -- preferred, each stream has independent state
- `GLOBAL` -- single state object for all streams
- `LEGACY` -- backward compatibility, deprecated

### 2.2 State Lifecycle

1. Platform sends last saved state to source at sync start
2. Source reads from cursor position in state
3. Source emits `AirbyteStateMessage` after processing records/slices
4. Platform persists state message
5. Next sync: platform sends stored state back to source

**Checkpointing interval**: Configurable via `state_checkpoint_interval` (e.g., every 100 records) or after each stream slice.

### 2.3 DatetimeBasedCursor State

```python
# Internal state representation
@property
def state(self) -> Mapping[str, Any]:
    return {self.cursor_field: str(self._cursor_value)}

@state.setter
def state(self, value: Mapping[str, Any]):
    self._cursor_value = value[self.cursor_field]

# During record reading
def read_records(self, ...):
    for record in records:
        self._cursor_value = max(record[self.cursor_field], self._cursor_value)
        yield record
```

**Stored state example for DatetimeBasedCursor:**
```json
{"updated_at": "2026-03-27T10:00:00Z"}
```

**Stored state example for IncrementingCountCursor:**
```json
{"id": 54321}
```

### 2.4 Key Design Properties

- State is opaque to everything except the source -- only the source reads/writes it
- `lookback_window` (e.g., `P3D`) re-reads 3 days before cursor to catch late-arriving data
- `step` (e.g., `P30D`) partitions time range into windows for manageable API calls
- `cursor_granularity` (e.g., `PT1S`) prevents overlapping windows
- Records must be in ascending order by cursor_field for correct state tracking

---

## 3. Comparison & Mapping: Airbyte State -> Splunk KV Store Checkpoints

### 3.1 Conceptual Mapping

| Airbyte Concept | Splunk KV Store Equivalent |
|-----------------|---------------------------|
| `AirbyteStateMessage` | `ckpt.update(key, state)` call |
| `stream_state` dict | `state` dict passed to `update()` |
| `stream_descriptor.name` | Checkpoint `_key` (input stanza name) |
| `cursor_field` value | Field in state dict (e.g., `last_timestamp`) |
| `state_checkpoint_interval` | Save state every N records or end of collection |
| State persistence (platform DB) | KV Store collection |
| `lookback_window` | Must implement manually: subtract window from stored cursor |

### 3.2 Storing Airbyte-Style Cursor State in Splunk KV Store

Yes -- they map directly. Example implementation:

```python
class CheckpointManager:
    """Bridges Airbyte-style cursor state with Splunk KV Store."""

    def __init__(self, ckpt, input_name):
        self.ckpt = ckpt
        self.input_name = input_name

    def get_cursor_value(self, cursor_field, default=None):
        """Get the last saved cursor value (like Airbyte stream_state)."""
        state = self.ckpt.get(self.input_name)
        if state and cursor_field in state:
            return state[cursor_field]
        return default

    def save_cursor_value(self, cursor_field, value, extra_metadata=None):
        """Save cursor value (like Airbyte emitting AirbyteStateMessage)."""
        state = {cursor_field: value}
        if extra_metadata:
            state.update(extra_metadata)
        state["_updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.ckpt.update(self.input_name, state)

    def apply_lookback(self, cursor_value, lookback_seconds):
        """Subtract lookback window from cursor (like Airbyte lookback_window)."""
        dt = datetime.fromisoformat(cursor_value.replace("Z", "+00:00"))
        adjusted = dt - timedelta(seconds=lookback_seconds)
        return adjusted.strftime("%Y-%m-%dT%H:%M:%SZ")
```

### 3.3 Best Practice for "Last Collected Up To X"

```python
def collect_incremental(helper, event_writer, api_client):
    input_name = helper.get_input_name()
    checkpoint_field = helper.get_arg("checkpoint_field")  # e.g., "updated_at"
    lookback_seconds = int(helper.get_arg("lookback_window") or 0)

    # 1. Load checkpoint
    state = helper.get_check_point(input_name)
    cursor_value = state.get(checkpoint_field) if state else None

    # 2. Apply lookback window (catch late-arriving data)
    if cursor_value and lookback_seconds:
        query_from = apply_lookback(cursor_value, lookback_seconds)
    else:
        query_from = cursor_value or helper.get_arg("start_datetime")

    # 3. Fetch records (with pagination)
    max_cursor = cursor_value  # Track high-water mark
    for page in api_client.paginate(since=query_from):
        for record in page:
            # Write event to Splunk
            event_writer.write_event(smi.Event(
                data=json.dumps(record, default=str),
                index=helper.get_arg("index"),
                sourcetype=helper.get_arg("sourcetype"),
            ))
            # Update high-water mark
            record_cursor = record.get(checkpoint_field)
            if record_cursor and (not max_cursor or record_cursor > max_cursor):
                max_cursor = record_cursor

    # 4. Save checkpoint (only if we found new data)
    if max_cursor and max_cursor != cursor_value:
        helper.save_check_point(input_name, {
            checkpoint_field: max_cursor,
            "_collected_at": datetime.utcnow().isoformat() + "Z",
        })
```

**Key differences from Airbyte to account for:**
1. Splunk inputs run on a polling interval (every N seconds), not batch sync
2. State is saved at END of collection interval, not mid-stream (no mid-interval checkpointing unless you add it)
3. No automatic lookback_window -- must implement manually
4. No automatic time-windowing (step) -- must implement pagination yourself
5. Records may NOT be in order -- use `max()` to track high-water mark, not "last record"

---

## 4. dpath Library

### 4.1 Overview

**dpath** is a Python library for accessing/searching dicts via `/slashed/paths`, like a filesystem glob.

- **PyPI**: https://pypi.org/project/dpath/
- **GitHub**: https://github.com/dpath-maintainers/dpath-python
- **License**: MIT
- **Python**: >=3.7
- **Dependencies**: ZERO (pure Python, stdlib only -- uses `fnmatch`)
- **Size**: ~15 KB (single module)
- **Downloads**: ~1.1M/week
- **Latest**: 2.2.0 (June 2024)

### 4.2 Core API

```python
import dpath

data = {
    "response": {
        "data": {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        },
        "meta": {"page": 1, "total": 50}
    }
}

# GET a value by path
users = dpath.get(data, "/response/data/users")
# => [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

# GET with list-style path (avoids separator issues)
users = dpath.get(data, ["response", "data", "users"])

# VALUES matching a glob pattern (wildcard support)
all_pages = dpath.values(data, "/response/*/page")
# => [1]   (only meta has "page")

# SEARCH returns matching paths+values
results = dpath.search(data, "/response/**/name")
# => nested dict with matching structure

# SET a value
dpath.set(data, "/response/meta/page", 2)

# NEW creates missing intermediate keys
dpath.new(data, "/response/meta/next_cursor", "abc123")

# DELETE a path
dpath.delete(data, "/response/meta/total")

# MERGE (deep)
dpath.merge(dest_dict, src_dict)
```

**Separator**: Default `/`, configurable: `dpath.get(data, "response.data.users", separator=".")`

**Wildcards**: `*` matches one level, `**` matches multiple levels (recursive glob).

**Filtering**: `dpath.values(data, "/users/*", afilter=lambda x: x.get("active"))`

### 4.3 Airbyte's Use of dpath

Airbyte's `DpathExtractor` uses dpath internally for record extraction from API responses:

```yaml
# Airbyte manifest.yaml
record_selector:
  type: RecordSelector
  extractor:
    type: DpathExtractor
    field_path: ["data", "results"]    # => dpath.get(response, ["data", "results"])
```

This is why dpath is relevant: it is the SAME extraction mechanism Airbyte uses, so adopting it in our Splunk connector gives direct compatibility with Airbyte field_path patterns.

### 4.4 Is dpath Bundled in Splunk Add-ons?

**No.** dpath is NOT part of solnlib, splunktaucclib, or any standard Splunk library. It must be explicitly added to `package/lib/requirements.txt` and vendored during `ucc-gen build`.

To include it:
```
# package/lib/requirements.txt
dpath>=2.1.0,<3.0.0
```

### 4.5 Alternatives Comparison

| Library | Path Syntax | Dependencies | Size | Wildcards | Set Values | Best For |
|---------|-------------|-------------|------|-----------|------------|----------|
| **dpath** | `/a/b/c` or `["a","b","c"]` | None | ~15 KB | Yes (`*`, `**`) | Yes | Simple path access + globs |
| **jmespath** | `a.b.c[*].name` | None | ~30 KB | Yes | No (read-only) | Query + reshape + filter |
| **glom** | `a.b.c` or tuple specs | boltons (~200 KB) | ~60 KB | No | Yes (`assign`) | Complex restructuring |
| **jsonpath-ng** | `$.a.b.c[*]` | ply (~100 KB) | ~40 KB | Yes | No | XPath-style queries |

### 4.6 Recommendation for Our Connector

**Use dpath** because:
1. Zero dependencies -- critical for Splunk add-on size constraints
2. Direct compatibility with Airbyte DpathExtractor field_path format
3. List-style paths (`["data", "results"]`) match Airbyte manifest syntax exactly
4. Supports wildcards for flexible extraction (`["data", "*", "records"]`)
5. Tiny footprint (~15 KB)
6. Write support (`dpath.new`, `dpath.set`) useful for response transformation

**Alternative consideration**: If we later need JMESPath-style filtering/projection (e.g., `users[?status=='active'].name`), we could add jmespath as a secondary option. But dpath covers the primary record extraction use case.

---

## Sources

- [Splunk Solutions SDK - checkpointer.py](https://splunk.github.io/addonfactory-solutions-library-python/modular_input/checkpointer/)
- [GitHub: cb-defense-splunk-app checkpointer.py](https://github.com/carbonblack/cb-defense-splunk-app/blob/master/TA-Cb_Defense/bin/ta_cb_defense/solnlib/modular_input/checkpointer.py)
- [GitHub: cb-defense-splunk-app base_modinput.py](https://github.com/carbonblack/cb-defense-splunk-app/blob/master/TA-Cb_Defense/bin/ta_cb_defense/modinput_wrapper/base_modinput.py)
- [Splunk Docs: Data checkpoints](https://docs.splunk.com/Documentation/Splunk/9.4.2/AdvancedDev/ModInputsCheckpoint)
- [Splunk Dev: About KV store collections](https://dev.splunk.com/enterprise/docs/developapps/manageknowledge/kvstore/aboutkvstorecollections)
- [Splunk Dev: Manage KV store with REST API](https://dev.splunk.com/enterprise/docs/developapps/manageknowledge/kvstore/usetherestapitomanagekv/)
- [Airbyte Protocol - State Messages](https://docs.airbyte.com/understanding-airbyte/airbyte-protocol)
- [Airbyte: Incremental Streams](https://docs.airbyte.com/connector-development/cdk-python/incremental-stream)
- [dpath on PyPI](https://pypi.org/project/dpath/)
- [dpath on GitHub](https://github.com/dpath-maintainers/dpath-python)
- [Snyk: solnlib checkpointer examples](https://snyk.io/advisor/python/solnlib/functions/solnlib.modular_input.checkpointer)
- [GitHub: splunk/addonfactory-ta-library-python Issue #38](https://github.com/splunk/addonfactory-ta-library-python/issues/38)
- [jmespath vs jsonpath-ng comparison](https://webscraping.fyi/lib/compare/python-jmespath-vs-python-jsonpath-ng/)
- [glom Tutorial](https://glom.readthedocs.io/en/latest/tutorial.html)
