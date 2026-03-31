---
name: Project goals and roadmap
description: Main goals for the Splunk dev container template project - UCC alignment, cross-platform, and first real app
type: project
---

Two main goals as of 2026-03-26:

1. **Improve the Splunk dev container template** — align with the UCC (Universal Configuration Console) framework and existing Splunk React UCC apps by examining their git repos. Ensure it works both inside the dev container and from a Mac/Linux host.

2. **Build a real first app** — a generic RESTful API connector using the UCC schema and SDK, so users can create customized inputs. Will leverage Airbyte and PyAirbyte where appropriate, especially for the connector framework.

**Why:** Template is functional but needs to follow established UCC conventions to produce production-quality Splunk apps. The RESTful API connector app is the first real use case — it needs UCC's configuration UI for user-defined custom inputs.

**How to apply:** When making template changes, check against UCC framework patterns. Ensure all tooling (Taskfile, Docker, scripts) works cross-platform (container + Mac + Linux). For the app, focus on UCC schema-driven configuration and SDK integration.
