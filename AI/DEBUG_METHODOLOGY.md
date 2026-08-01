# Debugging Method: Root-Cause Discipline

Companion to the artifact-specific docs in `AI/`. Those tell you *what* the tools
show. This one governs *how the investigation is run* — when a theory has earned
implementation effort, and when to stop.

Applies to: hangs, deadlocks, illegal-address traps, Xid faults, sanitizer
reports, and numerical mismatches where the defect is **not visible in the CuteDSL
source**. It does not apply to ordinary bugs where reading the code finds it.

These rules were written after an investigation spent hours building fixes on a
root-cause theory that was internally consistent, explained every observation, and
was wrong. The theory was checkable in minutes and never checked. Treat the
overhead below as cheap relative to that.

---

## The protocol, compressed

Re-read this list every time you are about to write a fix. The rest of the doc is
rationale; this is the contract.

1. **No patch before a checked prediction.** Write the theory block (Theory /
   Predicts / Falsified by / Cost to check / Status) into the ledger at
   `agent_space/ledger_<bug-slug>.md` and run the check. Cannot name a falsifier
   → not a theory → no patch.
2. **Trap-time evidence confirms only predictions registered in advance.** It
   never confirms a story assembled after looking at it.
3. **Every hex offset, register name, or line number you cite must be greppable
   verbatim** in an artifact saved under `agent_space/`, cited as `file:line`.
   Fails the grep → the claim is fabricated → delete the claim.
4. **Validating a fix:** wipe the compile cache; run fixed and unfixed builds
   N ≥ 10 each; run the perturbation control (a semantically neutral edit — if it
   also "fixes" the bug, your green runs mean nothing about mechanism).
5. **Two failed fixes on one theory, or three reconciliations to save it, kills
   it.** Restart from the evidence or escalate to a human.

---

## The one rule

**A root-cause theory earns implementation effort only after it has made a
prediction that was checked.**

Not "explains all observations" — predicts something not yet observed, cheap to
test, that would be *false* if the theory is wrong. Before writing a fix, write
down:

```
Theory:      <mechanism, one sentence>
Predicts:    <observation that must hold if true>
Falsified by: <observation that would kill it>
Cost to check: <minutes>
Status:      UNTESTED | CONFIRMED | FALSIFIED
```

If you cannot name a falsifier, you do not have a theory — you have a narrative.
Narratives are fine as *candidates*; they do not get a patch.

Illustration. Suppose the theory is that the compiler merged a plain SMEM address
onto a cluster-rank-encoded base, making the load invalid on non-zero-rank CTAs.
That predicts **every faulting CTA has non-zero rank** — minutes to check against
the trap logs, and a single rank-0 fault kills it. A theory of this shape can
absorb a great deal of patch effort before anyone thinks to run the check, because
each failed patch reads as "the fix didn't reach the merge" rather than as
evidence against the merge.

---

## Evidence tiers

Rank evidence by how much the defect could have corrupted it.

**Tier 1 — trustworthy.** Deterministic source-level facts. Reproducible
pass/fail behavior across repeated runs. Divergence against a reference
implementation at a specific tensor index.

**Tier 2 — usable, needs corroboration.** PTX (`CUTE_DSL_KEEP_PTX=1`), dumped SASS
(`CUTE_CUBIN_PATH`), shared-memory layout offsets, `cute.printf` traces. Real, but
one binary's codegen is not the kernel's semantics.

**Tier 3 — contaminated by definition.** Anything captured *at* a trap: register
values, faulting addresses, block/thread IDs, `CUTE_DSL_LINEINFO` attribution,
cuda-gdb backtraces after `CUDA_EXCEPTION_*`. The instruction that faults is
frequently not the instruction that is wrong, and the reported line is frequently
not the line that is wrong. Also Tier 3: `compute-sanitizer --tool=racecheck` on
raw TMA paths — see `AI/RACECHECK_TMA_HAZARD.md` for the known false positives.

Tier 3 evidence **generates** hypotheses. It confirms one only in a single,
narrow way: when a theory built from Tier 1/2 evidence predicted a *specific*
trap signature **in advance** — this CTA rank, this address pattern, this line —
and the trap matches. (The illustration in "The one rule" uses trap logs exactly
this way.) What Tier 3 can never do is originate confirmation post hoc: a story
assembled entirely from Tier 3 observations is at its most dangerous when it is
most coherent, because the same corruption that produced the fault also produced
the details that make the story fit.

Separate the columns explicitly in your notes:

| Observed (dump/log line) | Inferred (causal claim) |
|---|---|

Hallucinated mechanisms live in the right column while borrowing credibility from
the left. If the load-bearing claim has nothing in the left column, say so.

Make the left column auditable: **every hex offset, register name, or line number
quoted in the ledger or a report must be greppable verbatim in an artifact saved
under `agent_space/`** (trap log, dump, PTX, SASS), and cited as `file:line`.
Save the raw artifact *before* quoting from it. A detail that fails the grep is
fabricated — remove the claim, not just the citation. This check is mechanical;
run it on your own report before presenting it.

---

## Compile sensitivity: a green run proves almost nothing

FA4 JIT-compiles per configuration. Any edit — including a semantically neutral
one — reshuffles codegen. Consequences:

1. **A fix that works may have worked by perturbation.** There are two separate
   axes of nondeterminism, and they need separate tests:
   - *Runtime nondeterminism* (races, timing): run the fixed build **N ≥ 10**
     times, and the *unfixed* build the same number of times to establish the
     baseline failure rate. A bug that fires 1-in-3 will look fixed twice in a
     row. Note that repeated runs of unchanged source reuse the same cubin — they
     sample timing, not codegen.
   - *Codegen sensitivity* — run the **perturbation control**: apply a
     semantically neutral edit of similar size to the fix (a dead local, a
     reordered declaration) and test whether *it* also makes the bug vanish. If a
     no-op edit "fixes" the bug too, the real fix's green runs are worthless as
     evidence about mechanism — the defect is codegen-sensitive and the fix is,
     until proven otherwise, just another perturbation.
2. **Clear the cache when validating a fix.** `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`
   persists cubins at `/tmp/${USER}/flash_attention_cute_dsl_cache/`. A "fix
   confirmed" run that loaded a stale cubin confirms nothing. Wipe it, or run with
   the cache disabled, for every validation run.
3. **Config flags that flip a bug are not evidence about mechanism.** If toggling
   an unrelated flag changes the outcome, the honest reading is "codegen-sensitive
   defect," not "this flag interacts with my theory." The perturbation control
   above turns this suspicion into a direct test.
4. **Pin the compiler.** Record `nvidia-cutlass-dsl` version, `ptxas` version
   (`CUTE_DSL_PTXAS_PATH` if custom), and driver in the ledger. A miscompile
   theory is only testable against a fixed toolchain.

---

## Unfalsifiability tells

Stop and re-derive when a theory (yours or one handed to you) shows these:

- **"Explains every observation."** Real root causes usually leave one or two
  loose ends. Total closure on the first pass is a warning, not a result.
  (Closure earned by a *checked* prediction is exempt — the warning is about
  closure achieved by narration.)
- **A randomness escape hatch** — "the optimizer rolls the dice each compile,"
  "timing-dependent," "depends which op inherits it." These clauses make every
  future result confirmatory. Any theory that cannot lose is not doing work.
- **Confidence language with no test attached** — "root cause nailed,"
  "definitively." Fluency is free; a discriminating experiment is not.
- **Precision as credential.** Exact hex offsets and register names invite belief.
  Grep the actual dump for them. Half-right details stitched with invented causal
  glue is the characteristic failure shape.

---

## When a fix contradicts the theory, the theory is dead

If the working fix cannot plausibly act on the hypothesized mechanism — a barrier
change "fixing" an address-CSE bug, a padding change "fixing" a race — that is a
**falsification**, not an unexplained detail. Do not harden the story around it.
The fix and the theory are now two separate open questions.

Corollary: **a fix that works does not validate the theory it came from.** This is
the single most expensive error available here, because the reward signal
(test passes) arrives exactly when the reasoning is worst.

---

## Ablation: useful, and weaker than it feels

To probe mechanism, reduce the fix to the weakest primitive that still works —
`sync_warp` before a named barrier, one padding element before a full realloc.

What this licenses: "the stronger primitive's extra guarantees were not necessary
*in these binaries*."

What it does **not** license: "the mechanism is X." The weaker fix may still work
by perturbation (see above), and sufficiency is not mechanism. Ablation results
need the same N ≥ 10 repetition and cache hygiene as any other validation run.

Also look for **an existing control**: often a run from earlier in the
investigation already varies only the hypothesized trigger. Say plainly that it
was not designed as a control when you use it that way.

---

## Breaking a stuck investigation

The dominant failure is not misunderstanding CUDA — it is a session whose context
has accumulated in favor of the incumbent theory, where every new observation gets
read through it. Two interventions, in cost order:

**1. Fresh-context adversarial review (cheap, do this first).** Open a new session.
Paste the *evidence only* — dumps, repro, observations — with the theory and the
trajectory stripped out. Ask: what are the two or three candidate mechanisms, and
what is the cheapest experiment that discriminates between them? Models reliably
fix errors presented as external input while failing to fix the same errors in
their own output; re-asking the same session "are you sure?" does not work and
often makes the answer worse.

**2. Fan out at the commitment boundary (expensive, use sparingly).** When a theory
is about to consume real implementation effort, spawn 3–5 *isolated* branches from
the same evidence. Each returns only:

```
(hypothesis, cheapest discriminating experiment, predicted observation)
```

No patches. Do not let the branches see each other's output and do not have a model
judge between them — peer exchange produces conformity, and the most fluent
narrative wins a judged comparison regardless of correctness. **You** run the
predicted experiments; the hardware selects. Portfolio search is worth its cost
here only because this repo has an oracle: you can run the kernel.

Mechanically, for an agent: launch each branch as an **isolated subagent** whose
prompt contains only the ledger's *Observed* column and the repro command —
no theories, no investigation history, no sibling output. Role-playing the
branches inside one session is not fan-out; a single context generating "five
hypotheses" produces five variations of its incumbent theory and defeats the
entire point of isolation.

---

## Stop conditions

Escalate to a human, or restart search from the evidence, when any of these hold:

- Two fixes built on a theory have failed.
- The theory has survived only because each counter-observation got a new
  reconciliation. Count them; three is too many.
- The last three experiment cycles (edit-compile-run rounds that could have
  produced a discriminating result) checked no falsifiable prediction.
- The next step requires believing that trap-time evidence is accurate about
  a mechanism no Tier 1 or Tier 2 observation supports.

---

## Recording

Keep a hypothesis ledger at `agent_space/ledger_<bug-slug>.md` for the duration
of the investigation — one file per bug, appended as results land, alongside the
raw artifacts (trap logs, dumps) it cites. It is the artifact that makes "how
many rescues has this theory needed" answerable. Shape:

```markdown
# bwd hang, 2CTA hdim128 — ledger
Toolchain: nvidia-cutlass-dsl 4.5.2, ptxas 13.0, driver 580.xx
Repro: CUDA_VISIBLE_DEVICES=3 pytest tests/cute/test_flash_attn.py -k "..." (fails ~1 in 3)

## Evidence
| Observed (artifact, file:line) | Inferred (causal claim) |
|---|---|
| trap.log:44 — faulting CTA has rank 1 | base address not cluster-mapped? |
| trap.log:91 — faulting CTA has rank 0 | — |

## Theories
### T1: compiler CSE'd plain SMEM base onto mapa-encoded base
Predicts:      every faulting CTA has non-zero rank
Falsified by:  any rank-0 fault in the trap logs
Cost to check: 5 min (grep trap.log)
Status:        FALSIFIED — Evidence row 2. Five minutes of grep; do not patch.
Rescues:       0
```

In the final report or commit message:

- State unproven mechanisms **as hypotheses**, and name the experiment that would
  settle each one. Do not promote a plausible mechanism to a stated cause because
  the fix landed.
- Distinguish "fix validated" (N runs, cache cleared, baseline established) from
  "mechanism established" (prediction made and confirmed). These are different
  claims and usually only the first is true.
- **Record the wrong turns.** A report that presents only the final theory teaches
  the next reader — human or agent — that the answer was obvious, and destroys the
  information about which evidence was misleading. That information is the most
  reusable part of the investigation.