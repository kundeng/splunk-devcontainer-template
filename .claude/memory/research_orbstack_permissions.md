---
name: Research - OrbStack file permissions and UID mapping
description: How OrbStack handles file ownership on bind mounts, especially in Mac -> devcontainer -> Splunk container chain
type: research
date: 2026-03-27
---

# OrbStack File Permissions & UID Mapping for Bind Mounts

## Context: Our Setup

```
Mac host (UID 501)
  └─ OrbStack lightweight Linux VM (VirtioFS)
       └─ Devcontainer (vscode, UID 1000) — bind-mounts /workspace from Mac
            └─ Docker socket (DooD) — sibling containers
                 └─ Splunk container (splunk, UID 41812) — bind-mounts subdirs of /workspace
```

Key bind mounts from `docker-compose.yml`:
- `${LOCAL_WORKSPACE_FOLDER}/splunk/config/apps:/opt/splunk/dev-apps`
- `${LOCAL_WORKSPACE_FOLDER}/splunk/stage:/tmp/apps`
- `${LOCAL_WORKSPACE_FOLDER}/react:/opt/splunk/react`
- `${LOCAL_WORKSPACE_FOLDER}/ucc/output:/opt/splunk/ucc-apps`

---

## 1. OrbStack's Filesystem Layer

**OrbStack uses VirtioFS** (Apple's virtual filesystem for VMs) with custom dynamic caching and optimizations built on top. It is NOT a custom filesystem from scratch — it extends VirtioFS with proprietary caching that reduces per-call overhead by up to 10x, yielding 2-5x real-world speedups (75-95% of native macOS performance).

**Architecture**: OrbStack runs a single lightweight Linux VM (similar to WSL2) with a shared kernel. The Docker engine runs inside this VM. The engine's socket is forwarded to macOS so `docker` CLI works from Mac. Services are written in Swift, Go, Rust, and C.

**How it differs from Docker Desktop**:
- Docker Desktop historically used gRPC FUSE (osxfs), then switched to VirtioFS in 4.6+
- Docker Desktop added an xattr-based ownership layer on top (`com.docker.grpcfuse.ownership`)
- OrbStack uses VirtioFS directly with caching but does NOT implement the xattr ownership layer
- Both have an inherent performance cost vs. native Linux due to the macOS filesystem intermediary

---

## 2. UID Translation: What Actually Happens

### OrbStack's behavior (confirmed from GitHub issues and docs):

**Mac user (UID 501) creates a file on the project directory:**
- Inside ANY container, that file appears as **UID 0 (root)**
- OrbStack maps the macOS user to root inside the Linux VM

**Container process (any UID) creates a file on a bind-mounted Mac directory:**
- On the Mac filesystem, the file appears owned by the **Mac user** (UID 501)
- Inside the container, `ls -la` shows the file owned by whatever UID the process used
- BUT: the real on-disk owner is the Mac user, and OrbStack does NOT store the Linux UID anywhere

**`chown` inside a container on a bind-mounted path:**
- **Silently ignored** (or appears to succeed but has no effect)
- The file remains owned by the Mac user on macOS
- OrbStack does NOT use xattrs to store the intended Linux ownership
- This is the key difference from Docker Desktop

### Docker Desktop's behavior (for comparison):

**Docker Desktop stores Linux UID/GID in macOS xattrs:**
```
com.docker.grpcfuse.ownership: {"UID":41812,"GID":41812,"mode":755}
```
- `chown splunk:splunk /opt/splunk/dev-apps/myapp` would write the UID/GID to the xattr
- The macOS file remains owned by Mac user, but Docker's FUSE layer reads the xattr and presents the correct Linux ownership to containers
- This is why Docker Desktop "just works" for permission changes

### OrbStack does NOT use Linux user namespaces for this translation
The mapping is done at the VirtioFS layer, not via userns-remap.

---

## 3. OrbStack's "Magic" Permission Handling — How It Actually Works

OrbStack's reputation for "just working" comes from a specific design choice:

**All Mac-originated files appear as root (UID 0) inside containers.** Since most container processes either run as root or have been configured with appropriate permissions, this "just works" for the common case of reading files.

The mechanism:
1. VirtioFS presents macOS files to the Linux VM
2. OrbStack maps the macOS user's identity to UID 0 (root) in the VM
3. All containers share this VM, so they all see root-owned files on bind mounts
4. Containers can READ these files because root owns them (world-readable or root-readable)
5. Containers can WRITE to these files because the underlying macOS filesystem allows it (the Mac user owns them)

**What this means for our setup:**
- The devcontainer (UID 1000/vscode) sees bind-mounted files as root-owned but can read/write them
- The Splunk container (UID 41812) sees bind-mounted files as root-owned but can read/write them
- `chown` commands inside either container on bind-mounted paths are **no-ops**

**This is NOT**:
- A FUSE layer that presents files as owned by the requesting process (Docker Desktop does something closer to this)
- User namespace remapping
- ACL-based access

---

## 4. Multiple Containers with Different UIDs Writing to Same Directory

**Can container A (UID 1000) and container B (UID 41812) both read/write to the same bind-mounted directory?**

**YES** — because:
- Both containers see files as owned by root (UID 0) on bind-mounted Mac paths
- The actual write permission is controlled by macOS, which sees its own user as the owner
- Both containers write through VirtioFS, which authenticates as the Mac user
- Files created by either container appear owned by the Mac user on macOS
- Files created by either container appear as root (UID 0) inside containers

**The catch**: Neither container can `chown` files to their own UID on bind-mounted paths. If Splunk's entrypoint tries `chown -R splunk:splunk /opt/splunk/dev-apps`, it will silently fail or error out. The files remain root-owned inside the container.

**For Docker named volumes** (like `splunk-var` and `splunk-etc` in our setup): These live entirely in the Linux VM's filesystem, so normal Linux permissions apply — `chown` works as expected, and UIDs are real Linux UIDs.

---

## 5. POSIX ACLs

**OrbStack**: Added support for "CIFS extended attributes and POSIX ACLs" in a release update. This is for its CIFS-based file sharing mechanism. `setfacl` commands should work on OrbStack bind mounts.

**Docker Desktop**: Does NOT support POSIX ACLs on bind mounts. `setfacl` returns "Not supported" (known issue since docker/for-mac#3502).

**OrbStack is better than Docker Desktop here** — ACLs can be used as a workaround for multi-UID access patterns.

---

## 6. Container Restart Behavior

- **Bind mount permissions are stable across restarts** — they come from macOS and are not cached in any mutable state
- **Named volume permissions persist** — volumes live in the Linux VM's virtual disk (`data.img`, a sparse file up to 8TB)
- **No caching or state resets** affect permissions on restart
- If a container's entrypoint sets up permissions on named volumes, those persist. If it tries to set permissions on bind mounts, those are lost (since they were no-ops)

---

## 7. Docker-out-of-Docker (DooD) Specifics

In our setup, the devcontainer feature `docker-outside-of-docker:1` shares the host Docker socket. This means:

**Sibling container architecture:**
```
OrbStack VM
  ├─ Devcontainer (created by OrbStack, bind-mounts Mac /workspace)
  └─ Splunk container (created by docker compose from inside devcontainer)
       └─ bind-mounts are resolved by the HOST Docker daemon
```

**Critical path issue**: When `docker-compose.yml` specifies:
```yaml
volumes:
  - ${LOCAL_WORKSPACE_FOLDER:-..}/splunk/config/apps:/opt/splunk/dev-apps
```

The `LOCAL_WORKSPACE_FOLDER` must be the **Mac host path** (e.g., `/Users/username/project`), NOT the devcontainer path (`/workspace`). This is because the Docker daemon runs on the host (in OrbStack's VM), not inside the devcontainer.

Our `devcontainer.json` correctly handles this:
```json
"runArgs": ["--env", "LOCAL_WORKSPACE_FOLDER=${localWorkspaceFolder}"],
"remoteEnv": { "LOCAL_WORKSPACE_FOLDER": "${localWorkspaceFolder}" }
```

`${localWorkspaceFolder}` resolves to the Mac-side path, which OrbStack's Docker daemon can resolve via VirtioFS.

**Permission chain for DooD bind mounts:**
1. Mac user (UID 501) owns `/Users/username/project/splunk/config/apps/`
2. OrbStack's VirtioFS presents this to the Linux VM
3. Devcontainer sees it as root-owned at `/workspace/splunk/config/apps/`
4. Splunk sibling container sees it as root-owned at `/opt/splunk/dev-apps/`
5. Both containers can read/write because VirtioFS authenticates as the Mac user
6. Neither can meaningfully chown files on this path

---

## 8. OrbStack vs Docker Desktop: Concrete Comparison

| Aspect | OrbStack | Docker Desktop |
|--------|----------|---------------|
| Filesystem tech | VirtioFS + custom caching | VirtioFS (was gRPC FUSE) |
| Bind mount UID seen in container | root (0) for Mac-created files | Configurable via xattr |
| `chown` on bind mount | Silent no-op | Stored in xattr, visible to containers |
| Mac sees container-created files as | Mac user (UID 501) | Mac user (UID 501) |
| POSIX ACLs on bind mounts | Supported | NOT supported |
| Multiple container UIDs writing | All work (all go through Mac user) | All work (all go through Mac user) |
| Named volume permissions | Normal Linux behavior | Normal Linux behavior |
| Performance | 2-5x faster than Docker Desktop | Baseline |
| xattr ownership metadata | Not implemented | `com.docker.grpcfuse.ownership` |

---

## Implications for Our Project

### What works fine:
1. **Reading app source from Splunk container** — files appear root-owned, Splunk can read them
2. **Writing from devcontainer** — vscode user writes, Mac user owns, Splunk can read
3. **Multiple containers sharing bind mounts** — both devcontainer and Splunk work
4. **DooD bind mount paths** — `LOCAL_WORKSPACE_FOLDER` correctly uses Mac paths
5. **Named volumes** (`splunk-var`, `splunk-etc`) — normal Linux permissions, chown works

### What may cause issues:
1. **Splunk entrypoint chown** — if Splunk's startup scripts try to `chown -R splunk:splunk` on bind-mounted paths, they will fail or silently do nothing. This is why we use `--user splunk` and ensure the Splunk image's entrypoint can handle root-owned input directories.
2. **File mode bits** — `chmod` on bind-mounted files may work (depends on macOS file permissions) but the results may not match expectations
3. **Switching between OrbStack and Docker Desktop** — permission behavior differs, especially around `chown`. A setup that relies on Docker Desktop's xattr behavior will not work on OrbStack.
4. **CI/CD on Linux** — Linux Docker has NO permission translation layer. Files will have real Linux UIDs. Any setup that "works" on OrbStack due to the root-mapping may break on a Linux CI server.

### Recommendations:
1. **Never rely on `chown` for bind-mounted Mac paths** — design the setup so containers can work with root-owned files
2. **Use named volumes for Splunk's var/etc** (already done) — these have real Linux permissions
3. **Use `LOCAL_WORKSPACE_FOLDER` for sibling container mounts** (already done) — ensures Mac paths are used
4. **Test on Linux Docker** periodically — OrbStack's permissive behavior may mask real permission issues
5. **Consider POSIX ACLs** on OrbStack if fine-grained multi-UID access is needed (OrbStack supports them, Docker Desktop does not)
