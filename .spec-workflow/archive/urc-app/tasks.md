# Universal REST Client (URC) App — Tasks

## Phase 1: Scaffold & Build Pipeline

- [ ] `task-1.1` — Initialize UCC project structure under `ucc/` with globalConfig.json
- [ ] `task-1.2` — Create Taskfile primitives: `ucc:build`, `ucc:package`, `ucc:stage`
- [ ] `task-1.3` — Set up hybrid build pipeline (UCC generate + vendor Airbyte CDK)
- [ ] `task-1.4` — Verify `task ucc:build` produces valid add-on output

## Phase 2: UCC Configuration (globalConfig.json)

- [ ] `task-2.1` — Define Account entity (auth types: API key, Bearer, Basic, OAuth2)
- [ ] `task-2.2` — Define Input entity (endpoint URL, interval, index, sourcetype, manifest reference)
- [ ] `task-2.3` — Add Proxy and Logging configuration pages
- [ ] `task-2.4` — Validate generated UI renders correctly in Splunk

## Phase 3: Declarative Connector Engine

- [ ] `task-3.1` — Vendor Airbyte CDK declarative module (minimal footprint)
- [ ] `task-3.2` — Implement manifest loader (read YAML connector definitions)
- [ ] `task-3.3` — Implement stream reader (auth, pagination, record extraction)
- [ ] `task-3.4` — Wire connector engine to UCC modular input handler

## Phase 4: Checkpointing & State

- [ ] `task-4.1` — Implement KV Store checkpointer (using UCC's KVStoreCheckpointer API)
- [ ] `task-4.2` — Map Airbyte state messages to KV Store cursor entries
- [ ] `task-4.3` — Test checkpoint save/restore across input restarts

## Phase 5: Integration & Testing

- [ ] `task-5.1` — End-to-end test: configure input via UI, collect from sample API
- [ ] `task-5.2` — Test incremental collection (checkpoint resume)
- [ ] `task-5.3` — Package and install via `task ucc:stage`
- [ ] `task-5.4` — Add e2e test suite: `test-urc-lifecycle.sh`
