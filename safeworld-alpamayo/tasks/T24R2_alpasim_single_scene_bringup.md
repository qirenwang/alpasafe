# T24R2 — Official AlpaSim single-scene bring-up

Status: BLOCKED (runtime/infra) — source cloned & pinned; sim cannot run without sudo + large downloads
Date: 2026-06-21 (UTC)
Run dir: `artifacts/safeworld_alpasim_t24_retry/20260621T014528Z/`
Owner: Claude Code

## Goal

Bring up the minimum official AlpaSim Docker-Compose environment for one local NuRec scene and
produce one real completed rollout + metrics file.

## What was done

- Cloned `NVlabs/alpasim` and pinned to release **v2026.5**, commit
  `a1f05bb628f3d1d19d79d44188e836e9108f98c6`, into `/home/qiren/alpamayo1.5/external/alpasim`
  (source only, ~6 MB; NOT vendored into the SafeWorld git repo).
- Read AGENTS.md, CLAUDE.md, README.md, docs/ONBOARDING.md, docs/TUTORIAL.md; inspected
  `setup_local_env.sh` and `Dockerfile`; reviewed scene catalog `data/scenes/sim_scenes.csv`
  (917 rows) and `sim_suites.csv` (suite `public_2601` = 916 scenes — must NOT download whole suite).

## Why BLOCKED (exact)

AlpaSim runs the whole closed loop (renderer/NRE, physics, runtime, controller, driver, trafficsim)
as GPU **Docker-Compose microservices**; the NuRec renderer is only distributed as a CUDA 12.8
container. Per AlpaSim `docs/ONBOARDING.md` the host needs: Docker usable **without sudo**,
`docker-compose-plugin`, `docker-buildx-plugin`, Rust/cargo, uv>=0.9.17, CUDA>=12.8, NVIDIA CT.

In this environment:
1. **Docker daemon not accessible** — socket is `root:docker`; `qiren` is NOT in the `docker` group
   (only `mingyu` is). Adding to the group requires `sudo usermod -aG docker qiren` + re-login.
2. **`docker-compose-plugin` / `docker-buildx-plugin` not installed** — install needs `sudo apt`.
3. **`sudo` is password-gated** (no passwordless sudo) — automation cannot perform 1 or 2.
4. **`cargo`/Rust missing** — installable user-level via rustup (no sudo), but moot until 1-3.
5. **`uv sync --extra all`** for AlpaSim is a multi-GB download (torch, warp-lang, …) →
   classed as a large download; `SAFEWORLD_ALLOW_LARGE_DOWNLOADS` is UNSET.
6. **One NuRec scene** download (likely >1 GB) also requires the large-download flag.

Per operational rule 11, sudo/system changes are not performed; they are documented for the human.
Per the large-download rules, AlpaSim deps and the scene were not downloaded.

## Scene selection (recorded, not downloaded)

Selection policy result is documented in `outputs/audits/alpasim_nurec_capability_audit_after_install.md`.
No scene downloaded (gated as above).

## Decision

`BLOCKED_RUNTIME_INCOMPATIBILITY` (docker/sudo) + `BLOCKED_MISSING_ASSETS_OR_ACCESS`
(large-download authorization). Base simulator could not produce a real rollout, so Alpamayo
integration inside AlpaSim was not started. Standalone Alpamayo inference (outside AlpaSim) was run
instead — see T24R3/T24R4.
