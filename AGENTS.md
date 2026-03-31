# Splunk Dev Container Template

Docker-out-of-Docker development environment for building Splunk applications.

## Repository Layout

- **`Taskfile.yml`** — all automation (`task dev:up`, `task ucc:build`, `task ucc:appinspect`, etc.)
- **`.devcontainer/`** — devcontainer config, compose files (Python 3.13)
- **`splunk/`** — Dockerfile, entrypoint wrapper, Splunk container
- **`ucc/`** — UCC add-on source (globalConfig.json + package/lib/)
- **`react/`** — React UI workspace (optional, for custom UCC pages)
- **`docs/`** — detailed guides (tooling, UCC golden path, testing strategy)

## Key Workflows

### UCC Add-on Development
```bash
task ucc:init APP_NAME=my_addon    # scaffold globalConfig.json + package/
task ucc:build                      # ucc-gen build → ucc/output/
task ucc:link                       # symlink into dev Splunk
task ucc:dev                        # build + link + refresh (fast loop)
task ucc:appinspect                 # run Splunk AppInspect
```

### Build Pipeline
```
ucc/<app>/                 →  ucc/output/<app>/              →  /opt/splunk/etc/apps/<app>
(source: globalConfig +       (ucc-gen output: REST handlers,     (symlink via ucc:link,
 package/lib/)                 conf, UI, vendored deps)           bind-mounted from host)
```

### Testing Pattern
- **pytest + responses** for mocked HTTP (no network, no credentials needed)
- **unittest.mock** for time (rate limiters), Splunk SDK (passwords, KV Store)
- See `docs/testing-pure-python-engine.md` for full tutorial

## Docs

- **`docs/TOOLING-LESSON.md`** — React tooling stack explained ground-up
- **`docs/ucc-golden-path.md`** — UCC add-on handbook with lessons learned
- **`docs/testing-pure-python-engine.md`** — pytest/mocking tutorial with real examples

## Spec Workflow

This project uses the `spec-workflow` MCP for spec-driven development. Specs live in `.spec-workflow/specs/`.
