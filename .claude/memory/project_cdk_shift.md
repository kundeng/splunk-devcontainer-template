---
name: CDK shift — pip install replaces vendoring and custom engine
description: airbyte-cdk>=6 pip-installed; custom engine/components/manifest processing are legacy dead code; cdk_bridge.py is the active runtime
type: project
---

As of 2026-03-31, the URC app has shifted to pip-installing `airbyte-cdk>=6,<7` instead of vendoring or using the custom engine.

**Active code path:** `urc_app_input_helper.py` → `urc.cdk_bridge.collect()` → `ConcurrentDeclarativeSource` (from airbyte-cdk pip package)

**Dead code (legacy custom engine):** engine.py, all components/, interpolation.py, registry.py, manifest.py, validate.py, models.py, models_generated.py, extensions.py

**Why:** CDK handles everything the custom engine did (auth, pagination, extraction, transforms, cursors) plus concurrent execution, better error handling, and schema validation. No reason to maintain a parallel implementation.

**How to apply:** Don't reference or build on the legacy engine code. All manifest execution goes through cdk_bridge.py. test_manifests.py needs rewiring from engine to cdk_bridge.
