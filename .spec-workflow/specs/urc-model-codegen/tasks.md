# Tasks — URC Model Code Generation

## Phase 1: Codegen Pipeline

- [x] 1. Create header.txt and extensions.py scaffolding files
  - Files: ucc/urc_app/schema/header.txt, ucc/urc_app/package/lib/urc/extensions.py
  - header.txt: DO NOT EDIT warning + regeneration instructions
  - extensions.py: documented subclass pattern with commented example
  - Purpose: Files that codegen depends on or that establish the extension pattern
  - _Requirements: R7, R8, R10_

- [x] 2. Add `urc:update-schema` Taskfile target
  - Files: Taskfile.yml
  - Downloads schema from Airbyte CDK repo, runs datamodel-codegen with all flags, runs ruff format
  - All flags in one canonical place: --output-model-type pydantic.BaseModel, --target-python-version 3.9, --enum-field-as-literal all, --use-schema-description, --field-constraints, --use-unique-items-as-set, --disable-timestamp, --custom-file-header-path
  - Purpose: Single command for deterministic regeneration
  - _Requirements: R1, R3, R4, R5, R6, R7, R9_

- [x] 3. Run codegen and rename models.py to models_generated.py
  - Files: ucc/urc_app/package/lib/urc/models_generated.py (new), ucc/urc_app/package/lib/urc/models.py (delete)
  - Run `task urc:update-schema` to generate models_generated.py
  - Delete old models.py
  - Verify: no TypeN enums, schema descriptions as docstrings, field constraints present, no timestamp in header
  - _Requirements: R4, R5, R6, R7, R8, R9_

- [x] 4. Update import sites (validate.py)
  - Files: ucc/urc_app/package/lib/urc/validate.py
  - Change `from urc.models import` to `from urc.models_generated import` (2 import sites: lines 7 and 61)
  - Run existing tests to verify backward compatibility
  - _Requirements: NF-3_

## Phase 2: CI + Verification

- [ ] 5. Add CI staleness check
  - Files: .github/workflows/ci.yml (create or modify)
  - New job: regenerate to temp file, diff against committed file, fail if different
  - _Requirements: R2_

- [x] 6. Verify idempotency and backward compat
  - Codegen is idempotent (same schema → same output, no timestamps)
  - Manifest processing (parse + resolve + propagate) works for all 5 test fixtures
  - TypeN enums remain (--enum-field-as-literal dropped — incompatible with Pydantic v1 anyOf discrimination)
  - Pydantic validation via pydantic.v1 compat layer has known limitations with anyOf unions in dev env (pydantic v2); works on Splunk (pydantic<2)
  - Schema descriptions preserved as docstrings, field constraints present, clean header
  - _Requirements: R1, NF-3 (partial — full validation requires pydantic<2 env)_
