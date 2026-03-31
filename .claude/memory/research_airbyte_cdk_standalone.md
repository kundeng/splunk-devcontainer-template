---
name: Airbyte CDK standalone usage research
description: Can airbyte-cdk be used as a standalone Python library inside a Splunk add-on? Dependency footprint, API surface, slimming options
type: reference
---

## airbyte-cdk on PyPI
- Package: `airbyte-cdk` v7.13.0, MIT license, Python 3.10-3.13
- Repo: github.com/airbytehq/airbyte-python-cdk

## Standalone Usage — CONFIRMED WORKING

```python
from airbyte_cdk.sources.declarative.concurrent_declarative_source import ConcurrentDeclarativeSource
src = ConcurrentDeclarativeSource(source_config=manifest_dict, config=user_config)
src.check(logger, config)  # -> AirbyteConnectionStatus
src.read(logger, config, catalog)  # -> Iterator[AirbyteMessage]
```

No Airbyte platform needed. Manifest dict + config dict in, records out.

## Dependency Footprint — BLOCKING for Splunk

~150 MB total installed. Heavy deps NOT needed for declarative REST:
- pandas (67 MB), numpy (38 MB), grpcio (17 MB), nltk (11 MB), google-cloud-secret-manager

Deps that ARE needed (~20 MB):
- requests, pydantic, PyYAML, Jinja2, dpath, jsonschema, isodate, backoff, cachetools, pyrate-limiter

No `[declarative-only]` extra exists. All 40 deps are required.

## Slimming Options

| Approach | Compat | Size | Effort |
|----------|--------|------|--------|
| Fork + strip deps | 100% | ~20 MB | Medium |
| Vendor declarative subtree only | 100% declarative | ~20 MB | Medium-high |
| Custom interpreter, same schema | 95-100% | ~5 MB | High |
| pip install --no-deps + selective | 100% risky | ~20 MB | Low but fragile |
