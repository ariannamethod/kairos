## LOG

### 2026-07-01 — Go split: organism/trainer in `roots/`, generic glue in `go-tools/`

- The Go organism (`roots/kairos.go`) needs ~15 companion `.go` files from molequla; only `kairos.go` had been transferred. Brought in the needed ones, scrubbed molequla→kairos, and split by whether they touch the Go `*GPT` struct:
  - `roots/` (`package main`): trainer + model-physics — `notorch_trainer.go` (66 `*GPT` refs), `metaweights_overlay.go`, `metaweights_seeding.go`, `cross_graze.go`, `gpu_forward.go`, `gpu_forward_stub.go`.
  - `go-tools/` (`package gotools`): generic cgo/glue (0 `*GPT` refs) — `cgo_notorch.go`, `cgo_aml.go`, `cgo_notorch_cpu.go`, `cgo_notorch_cuda.go`, `gpu_bindings_linux.go`, `gpu_bindings_stub.go`, `gpu_notorch_stub.go`, `spa_coherence.go`.
- Package boundary: 43 `go-tools` symbols exported (lowercase→Uppercase) + ~92 call sites in `roots/` prefixed `gotools.`; `NTNanGuard.check`→`Check`; `go.mod` added (`module github.com/ariannamethod/kairos`). `go-tools` has zero `*GPT`/main-package refs → no circular import.
- Duplicate trainer removed: `aml_trainer.go` not transferred; the two `if CFG.Trainer=="aml"` dispatch branches and the `CFG.Trainer` field/default/`--trainer` flag removed — notorch is the only trainer.
- cgo wired to the vendored engine (`ariannamethod/`, `libkairos`): darwin links `libkairos.dylib` + Accelerate + rpath; `cgo_aml.go` got the Accelerate/cblas header (dropped its inline `ariannamethod.c`). `linux-cpu` links the vendor `libkairos.so` (was system `-lnotorch`); `-lkairos` scoped to darwin so linux variants don't double-provide `nt_*`.
- Verified: `CGO_ENABLED=1 go build ./roots/` → 9.9 MB binary on darwin (neo); gofmt clean. Codex audit: boundary PASS, export PASS, KEEP PASS; its two FAILs (stale `CFG.Trainer`, linux double-link) fixed as above.
- All cgo variants now link the vendored `libkairos` (darwin `.dylib` + rpath; linux CPU and CUDA `.so` + rpath) — no more system `-lnotorch`/`-lnotorch_gpu`. OPEN (A40): the CUDA path additionally needs `libkairos` compiled `-DUSE_CUDA` (+ `notorch_cuda`, `-lcudart -lcublas`) on A40 — not verifiable from neo (darwin), as molequla's Go always built on A40.

### 2026-07-01 — vendor: fresh notorch + AML engine in `ariannamethod/`

- Created the `ariannamethod/` vendor at the repo root with the updated engine from upstream `notorch` (`b1959f4`) + `ariannamethod.ai` AML core (`9d80ac3`), replacing molequla's May-14 snapshot.
- notorch `4739 → 5354` lines: packed-GGUF `nt_qmatvec` (×7), op33 RRPRAM-lowrank, Chuck optimizer, Metal backend (`notorch_metal.{h,mm}`), GGUF loader (`gguf.{c,h}`), vision (`notorch_vision.h` + `stb_image.h`). AML core `8000 → 8423` lines (`ariannamethod.{c,h}` + CUDA). `notorch_simd.h` fresh (`632 → 661`); `notorch_simd_scalar.h` unchanged.
- Makefile rebranded to Kairos: builds `libkairos.{so,dylib}` from `ariannamethod.c + notorch.c` (kept the proven combined-lib build pattern + SIMD path; dropped the dangling `test_aml.c` target that was not vendored). Excluded the Python tier (`method.py`/`sentinel.py`/`__init__.py` — per Oleg) and build artifacts (`.o`/`.dylib`, now gitignored).
- Verified: `make` builds `libkairos.dylib` (317392 B) from the fresh sources (deprecation warnings only, no errors). Added repo `.gitignore` (artifacts + internal logs).

### 2026-07-01 — resnya: cut mortality (immortal organism) + DNA emission → shared kairos.txt

- Cut from all 4 ports (`roots/kairos.{c,go,rs,js}`): mitosis/division/child-spawn (`perform_mitosis`, `birth.json`, mitosis-lock, `MaxOrganisms` cap, `AcquireMitosisSlot`/`ReleaseMitosisLock`), hibernation/sleep (`perform_hibernation`, `should_hibernate`, `MarkHibernating`, `"sleeping"` status), apoptosis/death (`status='dead'` self-marker + peer `"dead"` handler), the overload-triggers that only fed divide/hibernate (`is_sustained_overload`, `relieve_overload`, syntropy CASE 6/7), mesh `messages(from_id,to_id,payload)` table + `swarm_log_message`/`LogMessage`, and the `parent_id` lineage column. Diff: 19 insertions / 606 deletions.
- KEEP intact: SwarmRegistry core (register/heartbeat/discover-peers), training-lock, DNA exchange, ontogenesis growth, corpus field, coherence/SPA, the chat-history `messages(ts,role,text)` table, the non-divide syntropy actions (amplify/boost/dampen/ground/explore/realign/steady). Kairos grows but never reproduces, sleeps, or dies.
- DNA emission repointed: all organisms now append their generated text to the one shared pool `kairos.txt` (C `dna_write` + Go `dnaWrite`), instead of per-element `../dna/output/<element>/` dirs. Each organism still READS its own element file via `--element` (`nonames_<element>.txt`).
- Verified: C `gcc` builds (140864 B); Go `gofmt` PARSE_OK + pure-Go typecheck shows zero resnya-orphans (remaining `undefined:` are pre-existing companion-file symbols Oleg did not transfer — `CrossField`/`blasDgemv`/`gpu*`); Rust zero residual + unused params underscored; JS `node --check` passes. Codex audit: PASS on dangling refs and build half-state, KEEP core intact.
- OPEN (next DNA-flow step): the consumption side (`dnaRead`/cross-graze) still points at the now-unfed `../dna/output` dirs — inert (runtime no-op, not a crash). Re-wiring it to read the shared pool is a small redesign (offset-tracked read of the growing file). Vestigial dead code (MaxOrganisms config, inheritedBurstHistory, SwarmInfo, overload-debug, birthConfig reader, idbPut) enumerated by Codex — non-breaking, follow-up cleanup.

### 2026-07-01 — scrub molequla brand + fix corpus paths

- Removed every "molequla" mention from the 4 ports (`roots/kairos.{c,go,rs,js}`): ckpt names → `kairos.ckpt` / `kairos_ckpt.json`, swarm dir → `~/.kairos/swarm` (consistent across c/go/rs), UI/log strings → `kairos`, JS class `MolequlaDB` → `KairosDB`, banners. Shared function identifiers kept. `grep -i molequla` over the 4 ports + README = 0.
- Realigned the `KAIROS.RS` box banner (`roots/kairos.rs:3297`).
- Default corpus path was `"nonames.txt"` (file absent) in all 4 ports → now `"kairos.txt"` (the main-Kairos container). Element runs unchanged: `--element earth/air/water/fire` → `nonames_<element>.txt` (files present). README logo → `assets/kairos.jpg` (present).
