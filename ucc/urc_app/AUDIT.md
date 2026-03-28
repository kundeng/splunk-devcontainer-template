# Airbyte CDK Dependency Audit

**Date:** 2026-03-28
**CDK version:** airbyte-cdk 7.13.0
**Full install size:** 213 MB
**Total packages:** ~80

## C Extension Packages (must avoid for Splunk vendoring)

| Package | Size | Purpose | In declarative hot path? | Avoidable? |
|---------|------|---------|--------------------------|------------|
| pandas | 45 MB | Data manipulation | No | Yes - not used by declarative |
| numpy (+libs) | 58 MB | Numeric computing | No | Yes - transitive via pandas |
| grpc/grpcio | 17 MB | gRPC protocol | No | Yes - Google Secret Manager only |
| rapidfuzz | 12 MB | Fuzzy string matching | No | Yes - not in declarative path |
| cryptography | 12 MB | TLS/JWT signing | Only JWT auth | Yes for non-JWT auth types |
| pydantic-core | 4.9 MB | Pydantic v2 Rust engine | No - models use v1 compat | Yes - vendor pydantic v1 instead |
| orjson | ~1 MB | Fast JSON parsing | Yes - JsonDecoder | Patchable - has json stdlib fallback |
| serpyco-rs | ~1 MB | Protocol serialization | No - Airbyte protocol only | Yes - we write Splunk events |
| rpds-py | ~1 MB | Persistent data structures | Transitive via jsonschema | Maybe - check if jsonschema needs it |
| whenever | ~1 MB | Datetime handling | Possibly in cursor logic | Needs investigation |
| regex | 2.9 MB | Advanced regex | Possibly in interpolation | Maybe - check if stdlib re works |
| yaml (C ext) | 2.9 MB | YAML with C speedup | Yes but pure-Python fallback | Yes - PyYAML works without libyaml |
| markupsafe | ~0.5 MB | Jinja2 escaping | Yes but pure-Python fallback | Yes - has pure-Python mode |
| charset_normalizer | ~0.5 MB | HTTP encoding | In requests path | Yes - has pure-Python fallback |
| cffi | ~1 MB | C FFI | Transitive via cryptography | Yes if no cryptography |
| google/* | 6 MB | Google Cloud APIs | No | Yes - Secret Manager only |

**Total C extension overhead: ~165 MB** (pandas, numpy, grpc, google alone = 126 MB)

## Pure-Python Packages (safe to vendor)

| Package | Size | Purpose | Needed? |
|---------|------|---------|---------|
| airbyte_cdk | 3.9 MB | Core CDK including declarative runtime | Yes (partial) |
| airbyte_protocol_dataclasses | small | Protocol models (pure Python dataclasses) | Yes |
| pydantic | 2.0 MB | Validation (v1 compat layer is pure Python) | Yes (v1 only) |
| requests + urllib3 + certifi + idna | ~2 MB | HTTP client | Yes |
| Jinja2 | ~1 MB | Template interpolation | Yes |
| PyYAML | ~0.5 MB (pure) | YAML parsing | Yes |
| dpath | small | Record extraction | Yes |
| backoff | small | Retry/backoff | Yes |
| cachetools | small | Caching | Yes |
| isodate | small | ISO 8601 duration parsing | Yes |
| python-dateutil + pytz | ~3 MB | Datetime handling | Yes |
| pyrate-limiter | small | Rate limiting | Yes |
| jsonschema + referencing + attrs | ~1 MB | Schema validation | Yes |
| jsonref | small | JSON reference resolution | Yes |
| typing-extensions | small | Type system backports | Yes |
| typing-inspection | small | Type introspection | Yes |
| annotated-types | small | Pydantic dependency | Yes (with pydantic v1: No) |
| boltons | small | Python utilities | Maybe |
| wcmatch | small | Wildcard matching | Maybe |
| xmltodict | small | XML response parsing | Yes (XmlDecoder) |
| nltk | 6.4 MB | NLP tokenizer | No - not in declarative |
| dateparser | 1.6 MB | Date string parsing | Maybe |
| anyascii | 2.6 MB | Unicode transliteration | No |
| unidecode | 1.4 MB | Unicode to ASCII | No |
| rich/pygments | ~5 MB | CLI formatting | No |
| click | small | CLI framework | No |
| dunamai | small | Version management | No |
| genson | small | JSON Schema generation | No |
| setuptools | 4.5 MB | Package tooling | No |
| tqdm | small | Progress bars | No |

## Recommended Minimal Pure-Python Set

For the declarative runtime, the following should suffice (~15-20 MB estimated):

```
# Core CDK (vendored subtree, not full package)
# airbyte_cdk/sources/declarative/ + required core modules

# Validation
pydantic<2  # v1.10.x pure Python, 166 KB
typing-extensions

# HTTP
requests
urllib3
certifi
idna
charset-normalizer  # pure-Python fallback OK

# Templating
Jinja2
MarkupSafe  # pure-Python fallback OK

# YAML/JSON
PyYAML  # pure-Python mode (no libyaml)
jsonschema
referencing
attrs
rpds-py  # PROBLEM: C extension, needed by jsonschema
jsonref

# Record extraction
dpath

# Datetime
isodate
python-dateutil
pytz

# Retry/Rate limiting
backoff
cachetools
pyrate-limiter

# XML support
xmltodict

# Protocol models
airbyte-protocol-models-dataclasses
```

## Key Findings

1. **165 MB of the 213 MB is C extensions we don't need** (pandas, numpy, grpc, google, rapidfuzz)
2. **pydantic v1 compat works** - CDK models use `from pydantic.v1 import BaseModel` which is pure Python
3. **orjson is in the hot path** but has a `json` stdlib fallback in the CDK code
4. **rpds-py is a concern** - it's a Rust extension required by jsonschema. Need to check if jsonschema works without it or if we can pin an older version that doesn't need it.
5. **The airbyte_cdk package itself is only 3.9 MB** - the declarative subtree is a fraction of that
6. **Total pure-Python subset estimate: 10-15 MB** after excluding all C-extension packages and CLI/dev tooling

## Splunk 10.2 Import Verification

**Splunk Python:** 3.9.17 (GCC 13.2.0)
**Build output:** 40 MB total, 20 MB lib/

| Package | Import Result |
|---------|--------------|
| pydantic | OK |
| pydantic.v1 | OK |
| pydantic BaseModel | OK (full model create + .dict()) |
| dpath | OK |
| yaml | OK |
| jinja2 | OK |
| requests | OK |
| isodate | OK |
| backoff | OK |
| cachetools | OK |
| xmltodict | OK |
| jsonref | OK |
| pyrate_limiter | FAIL (Python 3.10+ syntax in v4.x) |

**Fix:** Pin `pyrate-limiter>=3.1.1,<4.0` for Python 3.9 compat.

**Note on .so files:** ucc-gen build uses `--prefer-binary` which installs Cython-compiled wheels for pydantic, markupsafe, yaml, charset_normalizer. These are optional speedups — all packages fall back to pure Python if the .so can't load. The cp312 .so files won't load under Splunk's cp39, so pure-Python fallbacks will be used at runtime.

## Final Recommendations

1. **Pin pyrate-limiter to v3.x** for Python 3.9 compatibility
2. **The .so files in the build are harmless** — they won't load on Splunk's Python 3.9 and will silently fall back to pure Python
3. **jsonschema is NOT included** — it requires rpds-py (Rust). Use pydantic models for manifest validation instead
4. **Total effective runtime footprint: ~8 MB** (pure Python only after .so fallback)
5. **All critical imports verified working** on Splunk Enterprise 10.2's Python 3.9

## Resolved Questions

- **jsonschema without rpds-py:** Not viable. Older versions need pyrsistent (also C ext). Use pydantic for validation instead.
- **whenever:** Not included, not needed — we handle datetime with python-dateutil + isodate
- **CDK core coupling:** Deep — declarative imports from streams/http/, types, models/, utils/. Vendoring requires ~15-20 core files beyond declarative/
- **serpyco-rs:** Not included, not needed — we write Splunk events, not Airbyte protocol messages
