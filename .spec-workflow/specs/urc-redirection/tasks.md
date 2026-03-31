# URC Redirection — CDK 6.x Pivot Cleanup

**STATUS: COMPLETE** (E2E CDK test deferred — requires running inside Splunk container or Taskfile wiring)

The project shifted from a hand-written custom engine (38 components) to pip-installed `airbyte-cdk>=6` on Splunk 10.2 / Python 3.13. This spec captures the reorg work needed to align the codebase, specs, docs, and memory with that reality.

## Tasks

- [x] 1. Archive superseded specs (urc-phase2, urc-app) → `.spec-workflow/archive/`
- [x] 2. Remove dead code from `package/lib/urc/` — engine.py, components/ (10 files), interpolation.py, registry.py, manifest.py, validate.py, models.py, models_generated.py, extensions.py
- [x] 3. Rewire `test_manifests.py` — import from `urc.cdk_bridge` instead of `urc.engine`
- [x] 4. Clean up stale memory files — delete `research_airbyte_cdk_standalone.md`, update `project_rest_connector_plan.md` and `project_urc_architecture.md`
- [x] 5. Run `task ucc:build` — build succeeds, 62 MB total / 41 MB lib after cleanup
- [x] 6. Run `task ucc:appinspect` — 0 failures, 0 errors, 113 passes, 15 info warnings
  - Fixed: added manifest_server, dateparser_cli, test dirs to post-build cleanup
- [ ] 7. E2E CDK test — `test_manifests.py --live jsonplaceholder` (needs CDK in PYTHONPATH via Taskfile)
- [x] 8. Commit and push reorg to origin
