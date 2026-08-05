# Debugging Method: Root-Cause Discipline

Companion to the artifact-specific docs in `AI/`. Those tell you *what* the tools
show; this one governs *how the investigation is run* — when a theory has earned
implementation effort, and when to stop.

Applies to: hangs, deadlocks, illegal-address traps, Xid faults, sanitizer
reports, and numerical mismatches where the defect is **not visible in the
CuteDSL source**. Not to ordinary bugs where reading the code finds it.

Written after a real investigation (anonymized) spent hours building fixes on a
theory that was internally consistent, explained every observation, and was
wrong — the falsifier was checkable in minutes and already sitting in the
session's own logs. Two successor theories followed; the last was adopted
*after* the working fix landed, survived every validation run, and fell only
when the fix was disassembled and contained nothing resembling its credited
mechanism. Three wrong mechanisms, one episode. The overhead below is cheap
relative to that.

---

## The protocol, compressed

Re-read this list before writing any fix. The rest of the doc is rationale;
this is the contract.

1. **No patch before a checked prediction.** Write the theory block (Theory /
   Predicts / Falsified by / Cost to check / Status) into
   `agent_space/ledger_<bug-slug>.md` and run the check — starting with evidence
   already captured; the falsifier is often already in your logs. No falsifier →
   not a theory → no patch.
2. **Trap-time evidence confirms only predictions registered in advance** —
   never a story assembled after looking at it.
3. **Every hex offset, register name, or line number you cite must be greppable
   verbatim** in an artifact saved under `agent_space/`, cited as `file:line`.
   Fails the grep → fabricated → delete the claim.
4. **Validating a fix:** wipe the compile cache; run fixed and unfixed builds
   N ≥ 10 each; run the perturbation control (a semantically neutral edit — if
   it also "fixes" the bug, green runs mean nothing about mechanism).
5. **Disassemble the fix before explaining it.** Diff normalized SASS of both
   builds; confirm the fix's hypothesized action exists in the binary at all.
   (Fixes have turned out to compile to a literal NOP.)
6. **Two failed fixes on one theory, or three reconciliations to save it, kills
   it.** Restart from the evidence or escalate.

---

## The one rule

**A root-cause theory earns implementation effort only after it has made a
prediction that was checked.**

Not "explains all observations" — predicts something not yet observed, cheap to
test, that would be *false* if the theory is wrong:

```
Theory:      <mechanism, one sentence>
Predicts:    <observation that must hold if true>
Falsified by: <observation that would kill it>
Cost to check: <minutes>
Status:      UNTESTED | CONFIRMED | FALSIFIED
```

If you cannot name a falsifier you have a narrative, not a theory. Narratives
are fine as *candidates*; they do not get a patch.

Illustration, from the motivating episode. Theory: the compiler merged a plain
SMEM address onto a cluster-rank-encoded base, making the load invalid on
non-zero-rank CTAs. That predicts **every faulting CTA has non-zero rank** —
minutes to check, one rank-0 fault kills it. The check was never run; two fixes
were built and failed, each failure reading as "the fix didn't reach the merge"
rather than as evidence against it. When the trap logs were finally examined,
**both captured traps were rank-0** — the falsifying data predated the first
fix attempt. Corollary: check predictions against evidence you already hold
before designing new experiments.

Caution: what transfers from this example is the failure **shape** — coherent
narrative, unchecked cheap falsifier, patch effort absorbing contrary evidence —
not the mechanisms. Address CSE, warp reconvergence, barrier asymmetry,
`sync_warp` fixes: none is an elevated prior for a new bug; reaching for one
because you read it here is availability bias. The discipline is domain-general —
the ledger example under "Recording" runs the same protocol on a plain
numerical mismatch.

---

## Evidence tiers

Rank evidence by how much the defect could have corrupted it.

**Tier 1 — trustworthy.** Deterministic source-level facts; reproducible
pass/fail across repeated runs; divergence against a reference implementation
at a specific tensor index.

**Tier 2 — usable, needs corroboration.** PTX (`CUTE_DSL_KEEP_PTX=1`), dumped
SASS (`CUTE_CUBIN_PATH`), shared-memory layout offsets, `cute.printf` traces.
Real, but one binary's codegen is not the kernel's semantics.

**Tier 3 — contaminated by definition.** Anything captured *at* a trap:
register values, faulting addresses, block/thread IDs, `CUTE_DSL_LINEINFO`
attribution, cuda-gdb backtraces after `CUDA_EXCEPTION_*`. The faulting
instruction is frequently not the wrong instruction, and the reported line not
the wrong line. Also Tier 3: `compute-sanitizer --tool=racecheck` on raw TMA
paths — see `AI/RACECHECK_TMA_HAZARD.md` for the known false positives.

Tier 3 **generates** hypotheses. It confirms one only when a theory built from
Tier 1/2 evidence predicted a *specific* trap signature in advance and the trap
matches (the illustration above uses trap logs this way). It never originates
confirmation post hoc: a story assembled from Tier 3 alone is most dangerous
when most coherent, because the corruption that produced the fault also
produced the details that make it fit.

Separate the columns in your notes:

| Observed (artifact, file:line) | Inferred (causal claim) |
|---|---|

Hallucinated mechanisms live in the right column borrowing credibility from
the left; if the load-bearing claim has nothing on the left, say so. Make the
left column auditable: every hex offset, register, or line number quoted must
be greppable verbatim in an artifact saved under `agent_space/`, cited as
`file:line` — save the artifact *before* quoting it. Fails the grep → remove
the claim, not just the citation. Run this check on your own report before
presenting it.

---

## Compile sensitivity: a green run proves almost nothing

FA4 JIT-compiles per configuration; any edit — even a semantically neutral one —
reshuffles codegen. Consequences:

1. **A fix may have worked by perturbation.** Two axes, two tests:
   - *Runtime nondeterminism:* run fixed and unfixed builds **N ≥ 10** each to
     establish the baseline failure rate — a 1-in-3 bug looks fixed twice in a
     row. Repeated runs of unchanged source reuse the same cubin: they sample
     timing, not codegen.
   - *Codegen sensitivity — the perturbation control:* apply a semantically
     neutral edit of similar size (a dead local, a reordered declaration). If
     it also "fixes" the bug, the real fix is, until proven otherwise, just
     another perturbation.
   - *Instrumentation is a perturbation too:* a `printf` that makes a hang
     vanish has located nothing — it has shown the defect is
     timing/codegen-sensitive, which makes both controls above mandatory.
2. **Clear the cache when validating.** `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`
   persists cubins at `/tmp/${USER}/flash_attention_cute_dsl_cache/`; a
   "confirmed" run that loaded a stale cubin confirms nothing.
3. **Config flags that flip a bug are not evidence about mechanism** — the
   honest reading is "codegen-sensitive defect." The perturbation control turns
   that suspicion into a test.
4. **Pin the toolchain.** Record `nvidia-cutlass-dsl`, `ptxas`
   (`CUTE_DSL_PTXAS_PATH` if custom), and driver versions in the ledger; a
   miscompile theory is only testable against a fixed toolchain.

---

## Unfalsifiability tells

Stop and re-derive when a theory (yours or one handed to you) shows:

- **"Explains every observation."** Real root causes leave loose ends; total
  closure on the first pass is a warning. (Closure earned by a checked
  prediction is exempt — the tell is closure by narration.)
- **A randomness escape hatch** — "the optimizer rolls the dice,"
  "timing-dependent," "depends which op inherits it." These make every future
  result confirmatory; a theory that cannot lose is not doing work.
- **Confidence language with no test attached** — "root cause nailed,"
  "definitively." Fluency is free; a discriminating experiment is not.
- **Precision as credential.** Exact hex offsets and register names invite
  belief — grep the dump for them. Half-right details stitched with invented
  causal glue is the characteristic failure shape.

---

## When a fix contradicts the theory, the theory is dead

If the working fix cannot plausibly act on the hypothesized mechanism — a
barrier change "fixing" an address-CSE bug, a padding change "fixing" a race —
that is a **falsification**, not an unexplained detail. The fix and the theory
are now two separate open questions.

Corollary: **a fix that works does not validate the theory it came from.** This
is the most expensive error available here, because the reward signal (test
passes) arrives exactly when the reasoning is worst.

---

## Ablation: useful, and weaker than it feels

To probe mechanism, reduce the fix to the weakest primitive that still works —
`sync_warp` before a named barrier, one padding element before a full realloc.
This licenses "the stronger primitive's guarantees were unnecessary *in these
binaries*" — not "the mechanism is X." Sufficiency is not mechanism, the weaker
fix may still work by perturbation, and ablations need the same N ≥ 10 and
cache hygiene as any validation run.

**Disassemble the fix before explaining it.** Dump SASS for both builds (a
FakeTensorMode compile needs no GPU memory and can be verified bit-identical to
the real compile), strip addresses/labels/lineinfo, diff. Does the fix's
hypothesized action appear in the binary at all? Is the diff small enough to
read end to end? In the motivating episode the weakest-primitive fix emitted no
synchronization instruction whatsoever — a NOP plus a reshaped ptxas
convergence region — and the accepted mechanism died on the spot, *after*
passing every validation run. A mechanism story about a fix nobody has
disassembled is a story about an imagined binary.

**Rule-implication cross-check.** If the mechanism implies a general rule
("pattern X requires Y"), search the repo for a site with X and no Y that runs
correctly. One healthy counterexample kills the rule, in minutes.

**Retrospective controls.** A run from earlier in the investigation may already
vary the hypothesized trigger — usable, but say it was not designed as a
control, and hold it to the standard: it must differ from the failing
configuration in **one variable**. Reinterpreting a multi-variable run as a
control is narrative-building.

---

## Breaking a stuck investigation

The dominant failure is not misunderstanding CUDA — it is a context that has
accumulated in favor of the incumbent theory and reads every new observation
through it. Two interventions, in cost order:

**1. Fresh-context adversarial review (cheap, do first).** Open a new session;
paste the *evidence only* — dumps, repro, observations — with the theory and
trajectory stripped out. Ask for the two or three candidate mechanisms and the
cheapest experiment that discriminates between them. Models reliably fix errors
presented as external input while failing to fix the same errors in their own
output; asking the same session "are you sure?" does not work.

**2. Fan out at the commitment boundary (expensive, use sparingly).** Before a
theory consumes real implementation effort, spawn 3–5 **isolated subagents**,
each given only the ledger's *Observed* column and the repro command — no
theories, no history, no sibling output. Each returns only
`(hypothesis, cheapest discriminating experiment, predicted observation)`; no
patches. Do not let branches see each other's output and do not have a model
judge between them — peer exchange produces conformity, and the most fluent
narrative wins a judged comparison regardless of correctness. **You** run the
experiments; the hardware selects. Role-playing the branches inside one session
is not fan-out — a single context produces five variations of its incumbent
theory.

---

## Stop conditions

Escalate to a human, or restart from the evidence, when any of these hold:

- Two fixes built on a theory have failed.
- The theory survives only by reconciling counter-observations. Count them;
  three is too many.
- The last three experiment cycles (edit-compile-run rounds that could have
  produced a discriminating result) checked no falsifiable prediction.
- The next step requires trusting trap-time evidence about a mechanism no
  Tier 1/2 observation supports.

---

## Recording

Keep the ledger at `agent_space/ledger_<bug-slug>.md` — one file per bug,
appended as results land, alongside the raw artifacts it cites. It is what
makes "how many rescues has this theory needed" answerable. Shape:

```markdown
# fp16 mismatch, local attention hdim64 — ledger
Toolchain: nvidia-cutlass-dsl 4.5.2, ptxas 13.0, driver 580.xx
Repro: CUDA_VISIBLE_DEVICES=3 pytest tests/cute/test_flash_attn.py -k "..." (deterministic, fails every run)

## Evidence
| Observed (artifact, file:line) | Inferred (causal claim) |
|---|---|
| diff.log:12 — first divergence at (b=0, h=2, q=191, d=17); q=191 is the last row of its m-block | tile-edge mask handling? |

## Theories
### T1: local-window mask off by one on the diagonal n-block
Predicts:      the set of divergent q-rows moves when n_block_size goes 128 → 64
Falsified by:  divergent-row set unchanged across n_block_size
Cost to check: 10 min (one recompile, diff the mismatch indices)
Status:        CONFIRMED — set shifted exactly with the tile edge (diff_n64.log:3)
Rescues:       0
```

(The illustration in "The one rule" shows this table catching a falsified
theory; this one shows a confirmation earned by a discriminating prediction.
Both cost minutes.)

In the final report or commit message:

- Lead with the two statuses stated separately: `Status: FIXED` (N runs, cache
  cleared, baseline established) and `Mechanism: ESTABLISHED` (confirming
  prediction cited) or `Mechanism: OPEN`. Usually only the first is true. A
  report may stay at `Mechanism: OPEN` indefinitely; promotion costs a checked
  prediction, not a landed fix.
- State unproven mechanisms **as hypotheses**, naming the experiment that would
  settle each one.
- **Record the wrong turns.** A report that presents only the final theory
  teaches the next reader the answer was obvious, and destroys the information
  about which evidence was misleading — the most reusable part of the
  investigation.
- **Audit the lesson itself.** Post-mortems can repeat the fallacy one level
  up — first drafts reliably do. Give the report the same fresh-context review
  as the investigation. And do not overcorrect into discarding an evidence
  class: trap-time data is insufficient alone, not useless.
