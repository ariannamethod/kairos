## LOG

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
