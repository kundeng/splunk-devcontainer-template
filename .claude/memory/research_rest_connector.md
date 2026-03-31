---
name: REST connector architecture research
description: Architecture for generic RESTful API connector using UCC SDK - auth, pagination, response parsing, credential storage
type: reference
---

## Architecture: Build natively with UCC SDK, NOT embed PyAirbyte

PyAirbyte has ~100+ transitive deps, subprocess model, conflicting state management. Instead, borrow Airbyte's declarative design patterns.

### Proposed globalConfig.json:
- **Account tab** (with table): OAuth entity supporting basic, oauth2 (auth code + client_credentials), api_key. All secrets encrypted.
- **Proxy tab**: predefined UCC tab
- **Logging tab**: predefined UCC tab
- **Single "REST API Input" service** with fields:
  - url, http_method (GET/POST/PUT), request_headers (JSON), request_body, request_parameters (JSON)
  - response_type (json/xml/text), response_record_path (JSONPath for record extraction)
  - pagination_type (none/offset/cursor/link_header/page_number) + pagination_config (JSON)
  - checkpoint_field (for incremental collection)
  - account (referenceName), interval, index, sourcetype

### Python structure:
```
bin/rest_api_input.py              # BaseModInput subclass
bin/rest_api_lib/
    collector.py                   # Orchestrator
    auth/{basic,oauth2,api_key}.py
    pagination/{offset,cursor,link_header,page_number}.py
    response/parser.py             # JSONPath via jsonpath_ng
    checkpoint.py                  # KVStore wrapper
```

### Key patterns:
- Auth handlers receive resolved account dict (already decrypted by UCC framework)
- Paginators implement iterator: has_next() / get_next_request_params()
- Response parser uses jsonpath_ng to extract records from user-configured path
- Checkpointing via BaseModInput.save_check_point()/get_check_point() (KVStore-backed)
- OAuth2 token refresh stored via solnlib.credentials.CredentialManager

### Where PyAirbyte fits:
- Design inspiration only (declarative YAML manifests, auth types, pagination strategies)
- Airbyte connector YAML manifests can serve as specification docs to translate into Splunk input configs
- NOT as a runtime dependency
