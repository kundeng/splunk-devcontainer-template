---
name: Splunk Test Agent research
description: AI-driven test framework using Claude Agent SDK + Playwright MCP + Splunk MCPs — two-phase explore/execute architecture, community MCP landscape assessment
type: project
---

Architecture proposal at `/workspace/docs/splunk-test-agent-proposal.md`. Too large for the devcontainer template — needs its own repo.

**Why:** Two-phase design: (1) AI explores and crystallizes into deterministic YAML plans, (2) dumb runner executes YAML with full audit trail. Same machinery works for testing AND oncall operations (probe Splunk from SME issue lists).

**Key finding:** No existing Splunk MCP has full admin write ops (input CRUD, conf management, app lifecycle). Official MCP is read-only search. Community servers (livehybrid, deslicer) are closest but need evaluation for extensibility.

**How to apply:** Phase 0 is evaluating livehybrid/splunk-mcp and deslicer/mcp-for-splunk codebases before building anything. The build-vs-extend decision is blocking.

**Open tension:** When MCP servers have workflow tools (deslicer's troubleshooting agents), the YAML-plan-vs-agent-loop distinction blurs. Need to resolve during design.
