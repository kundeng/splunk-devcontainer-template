# Splunk Test Agent — Architecture Proposal

**Status:** Draft / Research Phase
**Date:** 2026-04-03
**Scope:** Standalone project (too large for the devcontainer template repo)

---

## Problem Statement

Testing Splunk apps is painful:
- Splunk's DOM is hostile — deeply nested iframes, dynamic class names, shadow DOM, framework wrappers. Selectors break constantly.
- State is everywhere — KV stores, conf files, indexes, REST endpoints, search jobs. Setup/teardown is complex.
- Nobody writes E2E tests for full flows ("configure account, create stream, verify data lands") because it takes 200+ lines of Playwright boilerplate.
- Oncall operations (performance audits, parsing health checks, license monitoring) are manual SPL sessions with no reproducibility or audit trail.

## Core Insight: Two-Phase Architecture

The key insight is separating **exploration** (AI-driven, creative) from **execution** (deterministic, auditable):

```
PHASE 1: EXPLORATION (AI-driven, produces artifacts)
  Input:  Natural language prompt OR PDF of issues
  Engine: Claude Agent SDK + MCP servers
  Output: Concrete YAML test/probe plan

PHASE 2: EXECUTION (deterministic, no AI needed)
  Input:  YAML plan
  Engine: Dumb YAML interpreter + Playwright + REST client
  Output: step-by-step trace.json + JUnit XML + screenshots + summary.md
```

The AI's job is to **explore and crystallize** — navigate the UI, discover selectors, write SPL, and produce a YAML plan that a non-AI runner can execute repeatedly. The runner's job is to **execute and trace** — loop through steps, log every action, capture every assertion.

### Open Question: The Blurry Line

When an MCP server is powerful enough to encapsulate callable workflows (e.g., deslicer's "missing data troubleshooting" workflow that runs 10+ checks in sequence), the distinction between "YAML plan + dumb runner" and "agent loop calling MCP tools" becomes blurry. A sophisticated MCP server with workflow tools might eliminate the need for YAML entirely for some use cases. This tension needs resolution during design — see "Community MCP Assessment" below.

---

## YAML Plan Contract

Generated YAML plans must guarantee:

| Property | What it means |
|---|---|
| **Repeatable** | Same YAML + same Splunk state = same outcome. No fuzzy selectors — concrete CSS/accessibility refs, exact SPL, explicit wait conditions |
| **Auditable** | A human reads it and understands exactly what will happen. Every step has `name`, `purpose`, and concrete actions |
| **Deterministic** | No AI needed at execution time (best case). All selectors, values, assertions fully resolved |
| **Idempotent** | Setup/teardown ensures clean state. Explicit preconditions checked before run |
| **Versioned** | Header with `generated_by`, `explored_at`, `splunk_version`, `app_version` |
| **Self-documenting** | Each step explains WHY, not just WHAT |

---

## MCP Landscape (as of April 2026)

### Official / Splunk-Published

| Server | Source | Capabilities | Write Ops | Notes |
|---|---|---|---|---|
| [Official Splunk MCP](https://splunkbase.splunk.com/app/7931) (CiscoDevNet) | Splunk/Cisco, v1.1.0 GA | SPL search, index metadata, saved searches, knowledge objects | **No** — read-only, RBAC-enforced guardrails | Installs as Splunkbase app on SH/SHC. Streamable HTTP transport. Best for search verification |
| [splunk-mcp-server2](https://github.com/splunk/splunk-mcp-server2) | Splunk (unofficial) | Search with JSON/CSV/Markdown output, SPL validation | **No** — guardrails block destructive ops | Python + TypeScript. Good search UX with output format options. Docker/STDIO/SSE transport |

### Community

| Server | Source | Capabilities | Write Ops | Notes |
|---|---|---|---|---|
| [livehybrid/splunk-mcp](https://github.com/livehybrid/splunk-mcp) | Community (FastMCP) | Search, indexes, dashboards, KV store management | **KV store CRUD** (create/list/delete collections) | SSE/STDIO/REST transport. Closest to admin ops. Needs deeper evaluation for conf/input management |
| [deslicer/mcp-for-splunk](https://github.com/deslicer/mcp-for-splunk) | Community | 20+ tools, 16 resources (CIM data models), workflow agents | **Workflow-oriented** (troubleshooting agents) | 170+ tests. Has AI-powered troubleshooting workflows (e.g., missing data investigation in 60s). Most mature community server. Blurs the YAML-vs-agent-loop line |
| [jkosik/mcp-server-splunk](https://github.com/jkosik/mcp-server-splunk) | Community (Go) | SPL search | **No** | Lightweight Go implementation, STDIO + SSE |
| [magifd2/splunk-mcp-go](https://github.com/magifd2/splunk-mcp-go) | Community (Go) | SPL search via REST | **No** | Minimal Go implementation |
| [CDataSoftware/splunk-mcp-server-by-cdata](https://github.com/CDataSoftware/splunk-mcp-server-by-cdata) | CData | Read via JDBC | **No** (read-only; full CRUD in paid CData Connect AI) | Commercial angle |

### Browser Automation

| Server | Source | Capabilities | Notes |
|---|---|---|---|
| [Microsoft Playwright MCP](https://github.com/microsoft/playwright-mcp) | Microsoft (official) | Navigate, click, type, snapshot (accessibility tree), wait, codegen | Accessibility-first (not pixel-based). Deterministic element identification. Production quality |
| [executeautomation/mcp-playwright](https://github.com/executeautomation/mcp-playwright) | Community | Browser + API automation, screenshots | Alternative with more API testing features |

### Agent SDK

| Library | Source | Notes |
|---|---|---|
| [Claude Agent SDK (Python)](https://github.com/anthropics/claude-agent-sdk-python) | Anthropic, v0.1.48 | In-process MCP via `@tool` decorator. Hooks for audit logging (PreToolUse, PostToolUse, Stop). Langfuse/MLflow tracing integration |

### Gap Analysis

**What exists:** Read-only search (official + community), KV store CRUD (livehybrid), workflow agents (deslicer), browser automation (Playwright MCP).

**What's missing for a test agent:**
- Input CRUD (create/update/delete modular inputs, scripted inputs, etc.)
- Conf file management (read/write stanzas in app conf files)
- Account management (create/delete UCC accounts)
- App lifecycle (install, enable, disable, restart)
- System operations (restart splunkd, reload conf)

**Key question:** Can livehybrid or deslicer be extended to cover the admin gap, or do we build a new MCP? Need to evaluate their codebases for extensibility.

### TODO: Deeper Community MCP Evaluation

Before deciding build-vs-extend, we need to:

1. **Clone and read** livehybrid/splunk-mcp source — evaluate:
   - FastMCP patterns and extensibility
   - How tools are registered (can we add input/conf/app tools?)
   - Auth model (token vs user/pass, multi-instance support)
   - Test coverage and quality

2. **Clone and read** deslicer/mcp-for-splunk source — evaluate:
   - The 20+ tools — which ones overlap with what we need?
   - The workflow agent pattern — does it subsume our YAML runner?
   - The 170+ tests — maturity indicator
   - CIM data model resources — useful for oncall mode?

3. **Assess overlap** — create a matrix of needed capabilities vs. what each server provides

4. **Decide strategy:**
   - (A) Extend livehybrid with admin tools (they already have FastMCP + KV store)
   - (B) Extend deslicer (more mature, but more opinionated)
   - (C) Build our own using patterns from both
   - (D) Compose: use deslicer for search/analytics + build a thin admin MCP for writes

---

## Dual Use Case: Testing + Oncall Operations

The same machinery serves both purposes:

### Testing Mode
```
Explorer prompt: "Explore the URC app and generate test plans for CRUD lifecycle"
Output: tests/plans/urc-crud-lifecycle.yaml
Runner: deterministic executor producing JUnit XML + traces
```

### Oncall Operations Mode
```
Explorer input: oncall-issues-q1-2026.pdf (SME-identified Splunk Cloud issues)
Output: ops/plans/perf-audit-2026-04-03.yaml (SPL probes with thresholds)
Runner: same executor, produces findings report instead of pass/fail
```

Both produce auditable, reproducible traces. Both use the same MCP servers. The difference is the explorer's system prompt and the report format.

### Example: Oncall Performance Audit Plan

```yaml
meta:
  name: Performance Audit - Slow Search Times
  source: oncall-issues-q1-2026.pdf
  issue_refs: ["ONCALL-1234", "ONCALL-1267"]
  tags: [oncall, performance]

steps:
  - name: Check license usage
    purpose: "ONCALL-1234 reported license warnings"
    assert:
      - search:
          spl: |
            | rest /services/licenser/usage
            | eval pct=round(quota_bytes_used/quota_bytes*100,1)
          expect: { pct: "< 90" }
          save_result: license_trend.csv

  - name: Identify slow searches
    purpose: "ONCALL-1267 reported dashboard timeouts"
    assert:
      - search:
          spl: |
            index=_audit action=search info=completed
            | where total_run_time > 60
            | stats count, avg(total_run_time) as avg_rt,
                    max(total_run_time) as max_rt by user, search
            | sort -max_rt | head 20
          save_result: slow_searches.csv
          expect: { count: "< 50" }

  - name: Check parsing pipeline
    purpose: Verify no parsing errors or queue pressure
    assert:
      - search:
          spl: |
            index=_internal source=*metrics.log group=queue
              name=parsingqueue
            | timechart span=1h max(current_size_kb) as queue_kb
          expect: { queue_kb: "< 500" }
```

---

## Proposed Directory Structure (Standalone Project)

```
splunk-test-agent/
  mcp-servers/
    splunk-admin/                    # MCP server for admin operations
      pyproject.toml
      splunk_admin_mcp/
        server.py                    # FastMCP entry
        tools/
          inputs.py                  # Input CRUD
          conf.py                    # Conf management
          kvstore.py                 # KV store CRUD
          search.py                  # SPL execution
          apps.py                    # App lifecycle
          system.py                  # Restart, health
        client.py                    # Splunk REST wrapper
      tests/

  agent/
    explorer.py                      # Claude Agent SDK — exploration only
    profiles/
      test-discovery.md              # System prompt: generate test plans
      oncall-investigation.md        # System prompt: probe from issue list
      regression-capture.md          # System prompt: create regression test

  runner/
    runner.py                        # Deterministic YAML executor (no AI)
    actions.py                       # Action handlers (browser, REST, search)
    assertions.py                    # Assertion evaluators
    trace.py                         # Trace/audit logging
    report.py                        # JUnit XML + markdown generation

  schema/
    plan-schema.json                 # JSON Schema for YAML validation

  plans/                             # Committed YAML plans (reviewed by humans)
    smoke.yaml
    templates/                       # Reusable plan templates

  ops/                               # Oncall operation plans
    plans/
    issues/                          # Input PDFs / issue lists
    reports/                         # Generated findings

  reports/                           # Execution traces (gitignored)

  Taskfile.yml                       # Task runner
  pyproject.toml
  README.md
```

---

## Implementation Phases

| Phase | What | Depends On | Standalone Value |
|---|---|---|---|
| **0** | Deep-dive community MCP evaluation (livehybrid, deslicer) | — | Informed build-vs-extend decision |
| **1** | `mcp-splunk-admin` server (or extend community) | Phase 0 decision | Any Splunk dev/admin can use it as an MCP tool |
| **2** | Deterministic runner + YAML schema + trace format | Phase 1 | Runnable test plans without AI |
| **3** | Explorer agent (Claude Agent SDK + profiles) | Phase 1 + 2 | AI generates YAML plans from prompts or PDFs |
| **4** | Oncall operations mode | Phase 3 | Same runner + explorer, oncall-specific profiles |
| **5** | Template integration | All | Taskfile tasks, devcontainer config, docs |

Phase 0 is critical — no point building an MCP server that duplicates 80% of what deslicer already has.

---

## Key Design Tensions

**1. MCP workflows vs. YAML plans** — deslicer's "missing data troubleshooting" workflow runs 10+ checks autonomously inside the MCP server. If that pattern scales, the "YAML plan + dumb runner" layer becomes overhead rather than value. The right answer might be: YAML for **repeatable regression tests** (deterministic, committed to git, reviewed), MCP workflows for **ad-hoc investigation** (exploratory, interactive, oncall).

**2. "Non-AI runner" purity** — The goal is "best case, a non-AI runner can execute the plan." But for browser automation against Splunk's hostile DOM, a truly non-AI runner will break whenever Splunk updates their UI. A pragmatic middle ground: the runner is deterministic for REST/SPL assertions (90% of value), but uses Playwright MCP's accessibility-tree snapshots (not AI, but smarter than raw CSS selectors) for browser steps.

**3. Where the intelligence lives** — The explorer agent generates plans. The runner executes them. But when the MCP server itself encapsulates multi-step workflows as callable tools, intelligence moves into the MCP layer. This is fine — it just means Phase 0 (evaluating community MCPs) determines how much custom infrastructure we actually need to build.

---

## Open Questions

1. **Build vs. extend vs. compose** — Does deslicer/mcp-for-splunk or livehybrid/splunk-mcp cover enough of the admin gap that we should extend rather than build from scratch?

2. **Workflow MCP tools vs. YAML plans** — If an MCP server provides high-level workflow tools (e.g., "troubleshoot missing data"), does the YAML plan layer add value or just add indirection? When is the agent loop + MCP tools the right abstraction vs. a deterministic plan?

3. **Splunk version compatibility** — The admin REST API varies across Splunk versions (9.x, 10.x, Cloud). How do we handle this in the MCP server? Feature detection? Version-gated tools?

4. **Auth model** — Token auth (best for CI) vs. user/pass (best for dev). Support both? How does this interact with RBAC when the MCP server needs admin-level access?

5. **Scope of "deterministic"** — Some assertions require polling (wait for data to appear in index). The runner handles this with explicit timeouts, but is a poll loop truly deterministic? What about time-sensitive tests?

6. **Report format for oncall** — JUnit XML makes sense for CI test results. What format works for oncall findings? Markdown? PDF? Splunk dashboard?

---

## References

### Official Splunk
- [Splunk MCP Server (Splunkbase)](https://splunkbase.splunk.com/app/7931)
- [Splunk MCP Server tools reference](https://help.splunk.com/en/splunk-enterprise/mcp-server-for-splunk-platform/mcp-server-tools)
- [CiscoDevNet/Splunk-MCP-Server-official (GitHub)](https://github.com/CiscoDevNet/Splunk-MCP-Server-official)

### Community MCP Servers
- [splunk/splunk-mcp-server2 (GitHub)](https://github.com/splunk/splunk-mcp-server2) — Unofficial Splunk, Python+TS
- [livehybrid/splunk-mcp (GitHub)](https://github.com/livehybrid/splunk-mcp) — FastMCP, KV store CRUD
- [deslicer/mcp-for-splunk (GitHub)](https://github.com/deslicer/mcp-for-splunk) — 20+ tools, 170+ tests, workflow agents
- [jkosik/mcp-server-splunk (GitHub)](https://github.com/jkosik/mcp-server-splunk) — Go, lightweight

### Browser Automation
- [Microsoft Playwright MCP (GitHub)](https://github.com/microsoft/playwright-mcp) — Official, accessibility-tree based
- [executeautomation/mcp-playwright (GitHub)](https://github.com/executeautomation/mcp-playwright) — Community alternative

### Agent SDK
- [Claude Agent SDK Python (GitHub)](https://github.com/anthropics/claude-agent-sdk-python)
- [Claude Agent SDK MCP integration docs](https://platform.claude.com/docs/en/agent-sdk/mcp)
- [Claude Agent SDK hooks (audit logging)](https://platform.claude.com/docs/en/agent-sdk/hooks)
- [Langfuse observability for Claude Agent SDK](https://langfuse.com/integrations/frameworks/claude-agent-sdk)

### Background
- [Playwright Codegen (deterministic test recording)](https://playwright.dev/docs/codegen-intro)
- [Playwright AI Ecosystem 2026](https://testdino.com/blog/playwright-ai-ecosystem/)
- [deslicer blog: Unlock AI Workflows in Splunk](https://deslicer.se/posts/unlock-ai-workflows-in-splunk-with-an-mcp-server)
