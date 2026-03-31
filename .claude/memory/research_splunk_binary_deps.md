---
name: Splunk supports binary Python deps
description: Splunk's Python 3.9 can load C extensions (.so) — pydantic v2 works, "pure Python only" was wrong
type: project
---

Splunk's bundled Python 3.9 CAN load C extension `.so` files. Tested and confirmed:
- `_ctypes`, `_ssl` load natively
- `pydantic-core` 2.41.5 (Rust-compiled `.so`) loads and works with pydantic 2.12.5
- Requires correct platform wheel: `cp39-cp39-manylinux_2_17_x86_64`

**Why:** The earlier "pure Python only" claim in golden path docs was based on assumption, not testing. Splunk's Python for Scientific Computing add-on (MLTK) ships numpy/scipy/sklearn — all heavy C extensions — confirming binary deps are supported.

**How to apply:**
- Pydantic v2 migration is now viable — enables `--enum-field-as-literal all` (drops 116 TypeN enums), proper discriminated unions, `RootModel`, better validation
- When vendoring deps, use: `pip install --target lib/ --python-version 3.9 --only-binary :all: --platform manylinux2014_x86_64`
- Update golden path docs to remove "pure Python only" constraint
- AppInspect: Splunk Cloud allows binary deps (MLTK proves this)

### CDK 6.x on Splunk 10.2 (confirmed 2026-03-30)
- `airbyte-cdk>=6` requires Python 3.10+
- Splunk 10.2 ships Python 3.13 — confirmed CDK 6.x loads and runs correctly
- This unblocks full CDK vendoring (previously blocked by Splunk's Python 3.9)
- Vendor command for 3.10+: `pip install --target lib/ --python-version 3.10 --only-binary :all: --platform manylinux2014_x86_64 airbyte-cdk>=6`
