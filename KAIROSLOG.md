## LOG

### 2026-07-01 — BPE in the born-Kairos heart (K02) + internal plan gitignored

- The born-Kairos trainer and inference move off byte-level onto a fixed learn-once
  BPE (wall #2): `ariannamethod/kairos_train.c` learns `K_MERGES` merges once,
  encodes the corpus, and writes the `K02` checkpoint (magic `K02\n` + cfg[6] +
  `n_merges` + `MergeRule[]` + tensors) — `kairos_train.c:50,264`; `kairos.aml`
  reads the same `K02` + its merge table — `kairos.aml:73`. Vocab = 256 + `K_MERGES`
  (1024 → 1280). BPE-OOM malloc guard + hash headroom `1<<22` at `kairos_train.c:85-92`.
- `.gitignore` now also ignores the internal `KAIROS_PLAN.md` (alongside
  `LOCAL_STATE.md` / `PROJECT_LOG.md`) — a planning doc, not for commit.
- Verified this turn (tool): C + trainer `gcc -fsyntax-only` rc0; Go `gofmt -e` rc0;
  JS `node --check` rc0; `amlc kairos.aml` → success; K02 magic consistent
  trainer↔inference; DNA writes to `kairos.txt` = 0 in both ports (grep).

### 2026-07-01 — DNA emission reverted to per-element ecology; kairos.txt is the seed only

- Corrected a resnya mistake (`b661697`): the DNA emission had been repointed to append into the shared `kairos.txt`. That polluted the seed. `kairos.txt` is the **main-Kairos primordial dataset** (its container) and must never be a DNA sink; the four mini-Kairoses each have their own per-element file.
- Reverted the emission in both writing ports back to the molequla ecology (exact pre-`b661697` form): C `dna_write` (`roots/kairos.c:4809`) and Go `dnaWrite` (`roots/kairos.go:5553`) now write each organism's generated fragment to `../dna/output/<element>/gen_<ts>_<step>.txt` again. This also re-matches the consumption side — `dna_read` (`roots/kairos.c:4822`) / `dnaRead` read siblings' `../dna/output`, so emission and consumption point at the same place once more (closes the write/read split on the write side).
- `kairos.txt` stays the default corpus / seed (`roots/kairos.c:165`, `roots/kairos.go:238`); each mini still reads its own element file via `--element` → `nonames_<element>.txt`. Rust/JS ports never emitted DNA (only read the seed corpus), so they were untouched.
- Verified: `grep fopen("kairos.txt"` in kairos.c = 0, `grep OpenFile("kairos.txt"` in kairos.go = 0; emission now to `../dna/output` in both; C `gcc -fsyntax-only -DUSE_BLAS` rc=0; Go `gofmt -e` parse OK.

### 2026-07-01 — construction-audit fixes, wave 1: coherence wall on, colony leftovers cut

- A Codex construction (design) audit found the project is an architectural fork — two organisms (the Go/ports molequla stack with BPE/overlay/growth/DNA, and the new C/AML resonance heart with byte-level/no-overlay) that share no weights/tokenizer/lifecycle — plus P0–P2 problems. This is wave 1: the clear, fork-independent fixes.
- **P0 (coherence wall was gated off):** `CorpusLogitOverlay` default `false → true` (`roots/kairos.go:275`). The Q-metaweights overlay — self-fading by logit magnitude (`metaweights_overlay.go:16`, weightless→trained gate) — is now active by default, so the mini-Kairos speaks from zero and the birth-corpus stays clean instead of the wall being CLI-gated and off.
- **P2 (dead colony leftovers):** cut `MaxOrganisms` (field + init, the divide cap, read nowhere after the resnya) from `kairos.go`; cut the JS mitosis child-birth reader (`birthConfig` / `childOrganismId` / burst inheritance) from `kairos.js`, keeping `urlParams` for element selection.
- Verified: Go cgo build green, JS `node --check` OK.
- **STILL OPEN — the load-bearing builds, sequenced:** (1) resolve the fork canon (read: minis = Go organism, Kairos = the AML heart, shared BPE); (2) BPE in the C heart (replace byte-level, grab RRPRAM `resonance-bpe.c`); (3) Q-overlay coherence in the C heart; (4) the real live-dialogue DNA bus (A→B utterance chain into `kairos.txt`) + maturation metric + birth-when-mature; (5) path/corpus contract; (6) datasets (blocked on the 5 chats + recursion mixing). Level-2 (limpha, multi-weight) after the born single-weight personality.

### 2026-07-01 — browser front `index.html` + Rust `Cargo.toml` + full-pipeline smoke

- Transferred molequla's `index.html` → repo root (scrubbed molequla→kairos). Paths configured via `<base href="roots/">`: `<script src="kairos.js">` → `roots/kairos.js`, logo `<img src="../assets/kairos.jpg">` → `assets/kairos.jpg`, and the front JS `fetch("kairos.txt")` / `"nonames_<element>.txt"` → `roots/kairos.txt` / `roots/nonames_*.txt`.
- Added `Cargo.toml` (root) so the Rust port builds: `[[bin]] path = "roots/kairos.rs"`, deps `rusqlite`(bundled)/`serde`/`serde_json`/`rand`. `cargo check` → Finished (21 dead-code warnings, no errors).
- Full-pipeline smoke (tool-verified): C `gcc` 140864B · JS `node --check` · Rust `cargo check` Finished · Go `CGO_ENABLED=1 go build ./roots` 9.9 MB · trainer `kairos_train.c` loss `5.58→4.09`/10 steps · `kairos.aml` `amlc` compile + run · `index.html` paths resolve.
- Codex audit: index.html base/path resolution, no other relative paths, no molequla residue, Cargo.toml bin+deps — all PASS, no FAILs.
- NEXT: RunPod A40 — real cross-platform build (linux cgo / CUDA) + the first mini-Kairos training run (needs the 6-point brief + the 5 raw chats).

### 2026-07-01 — resonance heart: C notorch trainer + `kairos.aml` inference

- `ariannamethod/kairos_train.c` — from-scratch trainer for the Kairos resonance/Janus arch on the notorch tape + Chuck: `seq_embedding → per layer {parametric RMSNorm → Q/K/V + RoPE → content MHA + RRPRAM-lowrank (op33) blended by a per-head sigmoid gate → WO residual → RMSNorm → SwiGLU} → final RMSNorm → lm_head → cross_entropy`. Writes a `K01` checkpoint. v1 tokenizer = byte-level (vocab 256), dims via `-D` defines. Verified: compiles against `libkairos`; trains (loss `5.5651 → 3.8913` in 20 steps — start ≈ ln(256), real gradient).
- `kairos.aml` (root) — INFERENCE: loads `K01`, runs the SAME forward FORWARD-ONLY (all params `nt_tape_param_frozen`, arch byte-identical to the trainer), samples with the AML Dario field overlay (`am_apply_destiny/attention_to_logits`). Two faces of attention = Janus (content + RRPRAM). Verified: `amlc` compiles + runs a generation on the trainer checkpoint.
- Codex audit: `K01` save↔load order PASS, forward shapes PASS, Chuck slot-order PASS. Fixed its FAILs: (1) inference RRPRAM ran at the current context length while `wr` is packed for `k->T` (op33 derives rank from `wr->len/(H*(D+T))`) → now a fixed left-padded `k->T` window, so short prompts are structurally correct; (2) empty prompt → seed `\n` guard; (3) `alpha` relabeled — the gate is FIXED this version (frozen, 50/50 content/rrpram, like the Go frozen-gate pattern that keeps sigmoid off the tape); a learnable on-tape gate is a follow-up. Exit-only tensor leaks noted as a follow-up (harmless).
- NEXT grabs from molequla (analysis): BPE C tokenizer (RRPRAM `resonance-bpe.c`) to replace byte-level; the Q metaweights overlay (`q/postgpt_q.c`) as the PRIMARY coherence-from-zero (the AML Dario field is the secondary layer); growth-stages (`molequla.go` `GrowthStages`) for embryo→10M ontogenesis; SPA reseed; field `.soma` persistence; GGUF export for the level-2 DoE parliament.

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
