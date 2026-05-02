# Splunk Dev Container Template

Docker-out-of-Docker development environment for building Splunk applications.

## Repository Layout

- **`Taskfile.yml`** — all automation (`task dev:up`, `task ucc:build`, `task ucc:appinspect`, etc.)
- **`.devcontainer/`** — devcontainer config, compose files
- **`infra/`** — Dockerfile, entrypoint wrapper, Splunk config, deps
- **`apps/`** — App source (UCC add-ons, React apps)
- **`output/`** — Build output (gitignored)

## Key Workflows

### UCC Add-on Development
```bash
task ucc:build                      # ucc-gen build → output/
task ucc:dev                        # build + link + refresh (fast loop)
task ucc:appinspect                 # run Splunk AppInspect
task ucc:package                    # build + package to infra/deps/
```

### Build Pipeline
```
apps/<app>/                →  output/<app>/                 →  /opt/splunk/etc/apps/<app>
(source: globalConfig +       (ucc-gen output: REST handlers,     (symlink via ucc:link,
 package/lib/)                 conf, UI, vendored deps)           bind-mounted from host)
```

Post-build fixups:
1. Fragment merging — `package/default.d/*.conf` appended to UCC output
2. `python.required = 3.9` injection — for Splunk 10.2+ / AppInspect compliance
3. `local/` cleanup — removes dev-time state that fails AppInspect

### Splunk Lifecycle
```bash
task dev:up                         # start Splunk (skip-provision on restart)
task dev:refresh                    # reload configs (~2s)
task dev:restartd                   # restart splunkd (~10s)
task dev:clean                      # full reset (removes volumes)
```

### Testing Pattern
- **pytest + responses** for mocked HTTP
- **live API tests** with `@pytest.mark.live` (real HTTP, skippable in CI)
- Integration conftest overrides mock for live-marked tests
