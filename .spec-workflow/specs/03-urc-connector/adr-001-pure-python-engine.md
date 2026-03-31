# ADR-001: Pure Python Engine over Airbyte CDK

**Status:** Accepted  
**Date:** 2026-03-31  
**Decision:** Build a pure-Python declarative engine from the Airbyte schema instead of depending on `airbyte-cdk`.

## Context

URC (Universal REST Client) is a Splunk add-on that executes Airbyte declarative manifests to collect data from REST APIs. It must run on:

- Splunk Enterprise 9.x (Python 3.9) and 10.x (Python 3.13)
- Splunk Cloud heavy forwarders (unknown OS/glibc)
- Customer-managed infrastructure (any Linux distro)
- Splunkbase distribution (must pass AppInspect)

## Options Evaluated

| | A: Full CDK | B: CDK stripped | C: CDK + shipped Python | D: Pure Python engine |
|---|---|---|---|---|
| **Approach** | `pip install airbyte-cdk>=6` ship everything | Install CDK, strip heavy deps, patch broken imports | Ship own Python runtime + full CDK | Implement engine from Airbyte schema, pure Python |
| **Package size** | 187 MB | 62-71 MB | 200+ MB | ~2 MB |
| **AppInspect** | Passes (cloud mode) | Passes after cleanup | Unknown | Passes trivially |
| **Python 3.9** | No (CDK 6.x needs 3.10+) | No | Possibly (ship 3.13) | Yes |
| **Python 3.13** | Yes | Yes | Yes | Yes |
| **Portability** | No — .so files need matching glibc | No — same .so problem after stripping | No — shipped Python still loads .so against system glibc | **Yes — zero native deps** |
| **glibc dependency** | numpy (2.38), cryptography (2.33), pydantic_core (2.28), serpyco_rs, rpds | Same — stripped deps crash at import, kept deps need matching glibc | Same — .so files still dlopen system glibc | **None** |
| **Runtime reliability** | Full CDK, all features | Stubs hide silent failures (serialization, rate limiting, state) | Full CDK if Python matches | Explicit — unimplemented = clear error |
| **Manifest compatibility** | 100% Airbyte | ~80% (pandas/numpy/crypto features broken) | 100% Airbyte | ~99% (schema-driven, same manifest format) |
| **Maintenance** | Chase CDK upgrades, new strip/patch sites | Same + patch script fragility | Ship/maintain a Python runtime | Own the code, upgrade schema when needed |
| **Dev effort** | Low (pip install) | Medium (strip/patch/debug) | High (Python packaging) | Medium-high (~5300 lines to write) |
| **Splunk Cloud** | glibc unknown, will break | Same | Unlikely allowed | **Works everywhere** |

## Why Options A-C Don't Work

### The glibc problem is fundamental

CDK 6.x depends on native extensions (Rust and C):

| Package | Type | Min glibc | Can be stubbed? |
|---------|------|-----------|-----------------|
| `pydantic_core` | Rust | 2.28 | No — pydantic v2 requires it |
| `serpyco_rs` | Rust | varies | No — CDK protocol serialization |
| `numpy` | C | 2.38 (v1.x on py3.13) | No — pandas requires it |
| `pandas` | C | depends on numpy | No — ResponseToFileExtractor |
| `cryptography` | Rust | 2.33 | Partially — Splunk ships its own, but needs env vars |
| `rpds` | Rust | varies | No — jsonschema requires it |

These `.so` files are compiled against a specific glibc version. When `ld-linux` loads them, it checks `GLIBC_X.XX` version tags. If the target system's glibc is older, the import fails with:

```
ImportError: /lib64/libc.so.6: version `GLIBC_2.33' not found
```

This is not fixable by:
- Shipping a different Python (Python loads .so via system glibc)
- Using `--platform manylinux_2_17` wheels (numpy<2 has no py3.13 wheel)
- Setting `LD_LIBRARY_PATH` (only helps with libssl, not glibc itself)
- Stripping the packages (CDK eagerly imports everything at init)

### The stripping approach is a liability

CDK's `__init__.py` eagerly imports ~150 classes across all subsystems. Stripping packages and patching import sites:
- Creates 6+ try/except stubs that silently degrade functionality
- Breaks protocol serialization (serpyco_rs stubs return wrong types)
- Removes rate limiting protection (APIs get hammered)
- Loses per-partition state accuracy
- Breaks on every CDK upgrade (new eager imports appear)

### Shipping Python doesn't solve it

Even a bundled Python interpreter loads `.so` extensions via `dlopen()`, which goes through the system's dynamic linker (`ld-linux`). The linker checks glibc version symbols. There is no way to bypass this without also shipping glibc itself (which would break the system).

## Why Option D Works

The Airbyte declarative manifest format is defined by a JSON Schema (`declarative_component_schema.yaml`, 5127 lines). This schema is the contract — not the CDK code. A pure-Python engine that:

1. Parses the same YAML manifests
2. Validates against the same schema (via generated pydantic v1 models)
3. Implements the same component behaviors

...produces identical results without any native dependencies.

### Dependencies (all pure Python)

| Package | Purpose | Size |
|---------|---------|------|
| `pydantic<2` | Model validation (pure Python, `--no-binary`) | 1.4 MB |
| `requests` | HTTP client | 150 KB |
| `Jinja2` | Template interpolation | 500 KB |
| `PyYAML` | Manifest parsing | 200 KB |
| `dpath` | Record extraction (DpathExtractor) | 50 KB |
| `isodate` | ISO 8601 duration parsing | 30 KB |

Total: ~2.5 MB. Works on Python 3.9+, any OS, any glibc, any architecture.

### Implementation scope (updated 2026-03-31)

- **Runtime code:** 3,516 lines, 50 registered components
- **Generated models:** 7,479 lines (auto-generated from schema, pydantic v1)
- **Test suite:** 3,151 lines, 176 tests (0.19s execution, no network)
- **Total:** ~14,146 lines of pure Python
- **Priority coverage:** CRITICAL, HIGH, and MEDIUM gaps all closed; only LOW priority remains

### What's NOT supported

| Feature | Reason | Impact |
|---------|--------|--------|
| OAuth2 PKCE | Requires cryptographic code verifier | Rare for server-side connectors |
| JWT RS256/ES256 signing | Requires RSA/EC crypto | Can add via pure-Python RSA lib if needed |
| File-based connectors | Airbyte-specific pattern, not REST | Out of scope for REST connector |

## Decision

Use Option D: pure-Python engine built from the Airbyte declarative schema.

Pin the schema version. Generate pydantic v1 models with `datamodel-code-generator==0.25.9`. Implement runtime components in pure Python. Target Python 3.9+ for maximum Splunk version compatibility.

## Consequences

- Must maintain ~3,500 lines of runtime code (vs 0 with CDK)
- Schema updates require regenerating models + potentially updating runtime
- 100% portable across all Splunk deployments
- ~2 MB package instead of 187 MB
- AppInspect passes trivially (0 errors, 0 failures)
- No glibc, no .so, no platform-specific builds
- 176 automated tests validate every component (see `testing-strategy.md`)
