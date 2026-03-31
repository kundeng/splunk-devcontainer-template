> **STATUS: PARTIALLY SUPERSEDED** — CDK 6.x (commit 598385e7) ships its own models, replacing hand-written Pydantic classes. However, the codegen pipeline (`task urc:update-schema`) remains useful for schema-driven UI generation.

# Requirements: URC Model Code Generation & Pydantic Layer

## Introduction

The URC app generates Pydantic v1 models from Airbyte's `declarative_component_schema.yaml` (JSON Schema Draft 7, 109 component types) using `datamodel-code-generator`. The current generated output works but has significant gaps: no CI staleness check, no base class for cross-cutting concerns, ugly auto-generated enum names, a hand-edited import shim that will be lost on regeneration, no schema descriptions preserved, timestamp noise in diffs, and no automation script. This spec covers hardening the code generation pipeline, creating an extensible model layer with support for debugging/tracing, and establishing CI guardrails.

## Glossary

- **codegen**: `datamodel-code-generator` CLI tool that produces Pydantic models from JSON Schema
- **schema**: `declarative_component_schema.yaml` — Airbyte's canonical component schema (JSON Schema Draft 7)
- **models.py**: The generated Pydantic model file (`ucc/urc_app/package/lib/urc/models.py`, ~3,674 lines, 136 model classes)
- **discriminator enum**: The `TypeN` enum classes (`Type`, `Type2`...`Type117`) — single-value enums used as `type` discriminators for `anyOf` unions
- **base class**: A shared parent class all generated models inherit from, enabling cross-cutting behavior
- **cross-cutting concerns**: Behaviors applied uniformly to all models: debug logging, tracing, validation diagnostics, serialization conventions

## Requirements

### Requirement 1: Reproducible Code Generation Script

**User Story:** As a developer, I want a single `task urc:update-schema` command that downloads the schema and regenerates models deterministically, so that regeneration is never done ad-hoc with inconsistent flags.

#### Acceptance Criteria

1. WHEN `task urc:update-schema` is run, THE generator SHALL download the latest `declarative_component_schema.yaml` from the Airbyte CDK repo and write it to `ucc/urc_app/schema/`
2. WHEN `task urc:update-schema` is run, THE generator SHALL invoke `datamodel-codegen` with all flags defined in a single canonical source (the Taskfile target or a `generate.sh` wrapper)
3. WHEN regeneration completes, THE output SHALL be identical given the same schema input — no timestamps, no environment-dependent differences
4. WHEN a developer inspects the generated file, THE file header SHALL contain the exact `datamodel-codegen` command used to produce it (via `--enable-command-header`) so it can be reproduced manually

### Requirement 2: CI Staleness Guard

**User Story:** As a team, we want CI to fail if the committed `models.py` is out of sync with the schema, so that stale generated code is never merged.

#### Acceptance Criteria

1. WHEN a PR is opened that modifies files under `ucc/urc_app/`, THE CI pipeline SHALL run `datamodel-codegen --check` (or equivalent diff) to verify `models.py` matches the schema
2. WHEN the generated output differs from the committed file, THE CI step SHALL fail with a clear message indicating which file is stale
3. WHEN `models.py` and the schema are in sync, THE CI step SHALL pass silently

### Requirement 3: Correct Pydantic v1 Targeting

**User Story:** As a developer, I want the generated models to import from `pydantic` directly (not `pydantic.v1`), matching our `pydantic<2` runtime constraint, so that no hand-editing of imports is needed after regeneration.

#### Acceptance Criteria

1. WHEN codegen runs, THE output SHALL use `from pydantic import BaseModel, Field` — not the `pydantic.v1` try/except shim currently present
2. WHEN codegen runs, THE flags `--output-model-type pydantic.BaseModel` and `--target-python-version 3.9` SHALL be set explicitly
3. WHEN codegen runs, THE flag `--use-annotated` SHALL NOT be used (Pydantic v1 does not support `Annotated` field syntax)
4. WHEN a developer runs codegen, THE tool SHALL be invoked in an environment with `pydantic<2` installed to ensure v1-compatible output

### Requirement 4: Clean Discriminator Types (Eliminate TypeN Enums)

**User Story:** As a developer reading the generated code, I want discriminator fields to use `Literal["TypeName"]` instead of opaque `Type42` enum classes, so that the code is self-documenting and IDE-friendly.

#### Acceptance Criteria

1. WHEN codegen runs with `--enum-field-as-literal all`, THE output SHALL use `Literal["AddFields"]` instead of `class Type3(Enum): AddFields = 'AddFields'`
2. WHEN the flag is applied, THE 116 single-value `TypeN` enum classes SHALL be eliminated from the output
3. WHEN multi-value enums exist (e.g., `Algorithm`, `HttpMethod`, `Action`), THEY SHALL remain as proper `Enum` classes

### Requirement 5: Schema Descriptions Preserved as Docstrings

**User Story:** As a developer using IDE autocomplete, I want schema descriptions to appear as class docstrings on generated models, so that I can understand component semantics without opening the YAML.

#### Acceptance Criteria

1. WHEN codegen runs with `--use-schema-description`, THE generated model classes SHALL have docstrings derived from the schema's `description` field (502 descriptions available in the schema)
2. WHEN a model class has no schema description, IT SHALL have no docstring (not an empty one)

### Requirement 6: Noise-Free Diffs on Regeneration

**User Story:** As a developer reviewing a PR after schema regeneration, I want the diff to show only meaningful changes (new/modified types), not timestamp noise or formatting churn.

#### Acceptance Criteria

1. WHEN codegen runs, THE output SHALL NOT include a timestamp in the file header (via `--disable-timestamp`)
2. WHEN codegen runs, THE output SHALL be formatted consistently with the project's formatter (ruff) — either via `--formatters ruff` or a post-generation `ruff format` step
3. WHEN the schema has not changed, REGENERATION SHALL produce a byte-identical output file

### Requirement 7: Generated File Clearly Marked as Do-Not-Edit

**User Story:** As a developer, I want the generated file to be unmistakably marked so that no one hand-edits it.

#### Acceptance Criteria

1. WHEN codegen runs, THE file header SHALL include a "DO NOT EDIT" warning with instructions to run `task urc:update-schema` instead
2. WHEN codegen runs with `--custom-file-header`, THE header SHALL include the do-not-edit warning AND the codegen command (combined with `--enable-command-header` or baked into the custom header)

### Requirement 8: Generated File Named Distinctly

**User Story:** As a developer, I want the generated file to be obviously distinguishable from hand-written code by its filename.

#### Acceptance Criteria

1. WHEN codegen runs, THE output file SHALL be named with a `_generated` suffix (e.g., `models_generated.py`) or placed in a clearly labeled directory
2. WHEN the project imports from the generated file, THE import path SHALL reflect the generated nature (e.g., `from urc.models_generated import ...`)

### Requirement 9: Field Constraints Carried Through

**User Story:** As a developer, I want schema constraints (`minimum`, `pattern`, `uniqueItems`) to be enforced in the Pydantic models at validation time, not silently discarded.

#### Acceptance Criteria

1. WHEN codegen runs with `--field-constraints`, THE output SHALL include `Field(ge=0)` for `minimum: 0`, `Field(regex=...)` for `pattern:`, etc.
2. WHEN the schema defines `uniqueItems: true` on an array, THE generated field SHALL use `Set` (via `--use-unique-items-as-set`) or document why `List` is acceptable
3. WHEN constraints are present, THEY SHALL be validated by Pydantic at `parse_obj()` time, catching bad manifest data earlier

### Requirement 10: Subclass Extension Pattern Documented

**User Story:** As a developer extending URC with custom component types, I want a clear pattern for adding validators and behavior to models without editing generated files.

#### Acceptance Criteria

1. WHEN a developer needs to add a validator to a generated model, THE project SHALL provide an `extensions.py` (or similar) file with subclass examples
2. WHEN `extensions.py` defines a subclass of a generated model, IT SHALL be importable and usable as a drop-in replacement in the validation pipeline
3. WHEN the project documentation describes the extension pattern, IT SHALL warn against editing `models_generated.py` directly

## Non-Functional Requirements

### NF-1: Generation Performance

THE codegen step SHALL complete in under 30 seconds for the 197 KB schema file.

### NF-2: No New Runtime Dependencies

THE base class and all model-layer improvements SHALL use only `pydantic<2` and `typing-extensions` — no new runtime dependencies added to the Splunk package.

### NF-3: Backward Compatibility

THE new generated models SHALL pass all existing manifest validation tests (`test_manifests.py`) without changes to test fixtures or manifest YAML files.

## Out of Scope

- Migrating to Pydantic v2 (blocked by Splunk's Python 3.9 + pure-Python requirement)
- Splitting models into multiple files (`--module-split-mode`) — single file is appropriate for this schema size
- Custom Jinja2 templates — standard codegen flags should be sufficient
- `--watch` mode for local development — regeneration is infrequent (only on upstream schema updates)
- `allOf` class hierarchy changes — the schema uses `anyOf` exclusively
- `--type-mappings` — no custom format types in the schema
- Serialization round-tripping (`by_alias`, `.dict()`) — models are validated once and never serialized back
