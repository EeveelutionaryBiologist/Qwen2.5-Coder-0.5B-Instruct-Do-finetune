# Fine-tuning plan

Fine-tune the `Qwen2.5-Coder-Instruct` family for the [Do](../Do) NL→shell translator,
conditioned on the target shell (**bash / zsh / fish**), which Do auto-detects and
injects into the prompt.

- **Training target:** single RTX 4090 (24 GiB).
- **Deployment target:** the user's **CPU**. Do's [config.toml](../Do/config/config.toml)
  defaults to `use_gpu = false`, 4 threads, `temperature = 0.0`, `n_predict = 96`.
- **Delivery format:** GGUF, for Do's llama.cpp backend.
- **Reference:** upstream [NL2SH](https://github.com/westenfelder/NL2SH) shipped a
  `Qwen2.5-Coder-0.5B-Instruct-NL2SH`; their training script is vendored verbatim at
  [scripts/nl2sh_legacy/finetune.py](scripts/nl2sh_legacy/finetune.py).

**Thesis:** Do already ships stock **1.5B Q4_K_M**. The assumptions is that a *fine-tuned 0.5B
beats a stock 1.5B* (or at least performs equally) on Do's narrow task at ~3× the inference speed. That comparison is
the project's success condition, not an incidental benchmark.

---

## 1. Model size — train both

| | 0.5B | 1.5B |
|---|---|---|
| Params | 494,032,768 | 1,543,714,304 |
| Q4_K_M on disk | ~400 MB | ~1.1 GB (already in Do's `models/`) |
| CPU decode, 4 threads | ~25–40 tok/s | ~8–15 tok/s |
| Latency for a ~20-token command | < 1 s | 1.5–3 s |
| Role | low-latency default | quality option |

**0.5B — full FT, standard AdamW:** ~9 GB peak (2.0 fp32 master + 4.0 Adam `m`/`v` +
2.0 bf16 weights/grads + <1 activations @150 tok). ~15 GiB headroom.

**1.5B — does *not* fit naively.** Standard AdamW mixed precision:

| Component | Size |
|---|---|
| bf16 weights + grads | 6.18 GB |
| fp32 master weights | 6.18 GB |
| Adam `m` + `v` (fp32) | 12.35 GB |
| **Total static** | **24.7 GB = 23.0 GiB** |

Against ~23 GiB usable after CUDA context and fragmentation, that leaves *zero* room for
activations. **Fix: `optim="adamw_bnb_8bit"`** — 8-bit optimizer states drop `m`+`v` from
12.35 GB to 3.09 GB, total **15.4 GB = 14.4 GiB**, leaving ~8 GiB for batch and
activations. Well-established at this scale, no meaningful quality cost. (Adafactor gets
to ~11.5 GiB if ever needed, but it's fussier and unnecessary here.)

**Cost:** 1.5B is 3.13× the params, but 0.5B at 150-token sequences badly underutilizes a
4090, so expect ~2.5–3× step time. If 0.5B runs 1–2 h, budget 3–6 h. Both are overnight jobs.

The data pipeline (§4) is the expensive, shared part; a second training run costs a few
GPU-hours. Train both, ship both, let Do switch between them.

## 2. Regime — full fine-tune via TRL `SFTTrainer`

**Full FT, not LoRA/QLoRA.** At these sizes PEFT solves a problem we don't have, converges
worse for a real domain shift, and complicates checkpoint export.

**Stack:** `torch`, `transformers` (v5), `trl==1.13.*` (v1.13.0, 2026-09-10), `datasets`,
`accelerate`, `bitsandbytes` (for the 1.5B run), optional `wandb`. Pin the TRL minor —
they are actively pruning old APIs.

**Starting hyperparameters**, from the legacy script: `lr=1e-5`, effective batch 75,
`max_len=150`, bf16. Note upstream used those for the **3B** model.

**Three fixes over the legacy script:**
1. Completion-only loss — it masks only pad tokens, so it trains on the system prompt and
   the English instruction too.
2. Dynamic padding or packing, not `padding="max_length"` — most pairs are far under 150.
3. Fewer epochs — 10 epochs over 40k unverified pairs invites memorizing noise. Sweep 2–4.

## 3. Prompt — mirror Do exactly

Do's prompt module is **vendored** at
[scripts/vendor/do_prompt.py](scripts/vendor/do_prompt.py) as a byte-identical copy, because
the Do checkout is not guaranteed to exist on the training machine. Importing it across
repos would be cleaner but cannot be relied on there.

The risk vendoring introduces is staleness — rule 4 already changed once (made
shell-conditional), and a stale copy silently invalidates every rendered example. Two
guards, since a copy can no longer be correct by construction:

- `finetune_do.py` hashes the vendored file against the Do checkout whenever one is
  reachable and **refuses to train** on a mismatch (`--allow-prompt-drift` to override).
  On the training machine, with no checkout, it proceeds unchecked — that is the trade.
- Provenance (upstream commit, sha256, `PROMPT_VERSION`) is recorded in
  [scripts/vendor/PROVENANCE.md](scripts/vendor/PROVENANCE.md), and `PROMPT_VERSION` plus the
  file hash are written next to every checkpoint as `do_prompt.json`. A checkpoint is only
  valid for the prompt it was trained against.

> **Trap:** use TRL's `{"prompt", "completion"}` dataset format, **not**
> `assistant_only_loss=True`. Do's prompt carries 8 few-shot assistant turns; assistant-only
> masking trains on all nine, memorizing `ls -a` / `ps aux` tens of thousands of times.

`build()` makes `shell:`, `os:`, `cwd:` and `files here:` all optional — 16 possible shapes.
Sample those combinations to match Do's runtime distribution, or every real invocation with
a cwd and file listing is out-of-distribution.

## 4. Data — rejection-sampled distillation

### Why not train on ALFA gold directly

Measured over `data/example` (NL2SH-ALFA):

| Finding | Value |
|---|---|
| Train rows | 40,639 (`nl`,`bash`, **unverified**) |
| Test rows | 300 rows × 2 refs (`bash`/`bash2`) = the 600 "pairs" on the card |
| Chains with `;` vs `&&` | 1,370 (3.37%) vs 75 (0.18%) — **18:1 against Do's rule 5** |
| Parses identically under bash **and** zsh | 99.8% (1,500 sampled, 0 zsh-only failures) |
| Contains any fish-divergent construct | 3.51% — backticks alone are 2.47 pts of that |
| Genuine structural divergence | **0.92%** — bare `VAR=x` 0.82%, `do/done` 0.07%, `then/fi` 0.03% |
| Test `nl` also appearing in train | 3 (1.0%) — dedupe |
| Duplicate `nl` within train | 729 distinct / 2,480 rows — dedupe before labeling |

Two conclusions:

1. **ALFA gold was written to a different spec than Do's.** The 18:1 `;`-over-`&&` ratio
   contradicts rule 5 outright, and the same holds for rule 2 (unrequested flags) — exactly
   what the few-shot comments (`not ls -la`, `not ps aux | grep`) exist to suppress. Training
   on gold teaches the model to violate Do's own system prompt.
2. **zsh is nearly free; fish is the entire job**, and the divergent slice is far too small
   for random sampling to ever teach it. If ~96% of examples have a gold command identical
   across shells, the model minimizes loss perfectly by ignoring the `shell:` line — then
   emits bash when asked for fish, while the aggregate score still looks fine.

**Note:** syntax checking alone is a near-worthless filter. 99.8% of the corpus already
parses; `bash -n` / `fish -n` catch almost nothing. A syntactically perfect `ls -la` for
"list files sorted by size" sails through.

**Three taxonomy corrections, measured in the sandbox — each shrinks the fish surface:**

- **`export VAR=val` is valid fish.** fish ships a compatibility function at
  `/usr/share/fish/functions/export.fish`. It was in the divergence taxonomy; it does not
  belong there.
- **`VAR=x command` is valid fish** (prefix override, since fish 3.1) and behaves
  identically — verified, not assumed. Only a *bare* `VAR=x`, or `VAR=x; ...`, is rejected.
  That splits the 1.07% "leading assignment" bucket into 104 valid rows and 332 invalid
  ones, dropping true structural divergence to 0.82%.
- **Backticks are not a syntax error in fish.** They parse cleanly and silently yield the
  literal text instead of a substitution — tier iii, not tier i, so no syntax check will
  ever flag them. They are the single largest fish-divergent construct at 2.47%.

### The pipeline

Keep ALFA's human-authored instructions, discard its gold as a *target*, reuse it as an
*oracle*:

1. **Teacher proposes.** `Qwen2.5-Coder-7B-Instruct` generates *n* candidates per
   instruction under Do's exact prompt and GBNF grammar. Policy-conformant by construction,
   since the teacher is obeying the same system prompt Do ships.
2. **Execution arbitrates.** Run candidate and ALFA gold in the sandbox
   ([scripts/shellrun.py](scripts/shellrun.py)); keep the candidate only when
   `compare()` returns `MATCH`. The teacher is a *proposal distribution*, not an authority —
   best-of-*n* with a verifier comfortably exceeds greedy teacher accuracy, so the student
   is not capped at the teacher's ceiling. For syntax screening use
   `check_syntax`, never a raw `fish -n`: **fish 3.6 exits 0 on syntax errors**, so only
   stderr detects them.
3. **Shell labeling.** Three-tier classification: (i) structurally impossible in fish → must
   translate; (ii) provably identical across shells → agnostic; (iii) parses everywhere but
   *behaves* differently (backticks — the biggest one at 2.47%, unquoted `$var` word
   splitting, `**` recursion, glob-no-match). Tier iii needs execution, not regex: these
   parse cleanly in fish and quietly do the wrong thing.
   - **Agnostic majority (ii):** assign bash/zsh/fish uniformly at random, **one label per
     row**. Stops "fish" becoming a rare token correlated with odd output. Do *not* emit
     three copies of `ls -a` — that triples compute and reinforces label-ignoring.
   - **Divergent slice (i/iii):** teacher proposes fish/zsh candidates, verified against the
     bash gold's *output*. These become deliberate **contrastive pairs** — same `nl`,
     different shell labels, genuinely different gold commands. Highest-leverage item in this
     document: the model cannot minimize loss without attending to the shell line.

Constraints on the divergent slice: **breadth over depth** (enumerate a taxonomy — one-line
loops, conditionals, `set -x`, local vars, `math`, `$status`, prefix assignment, globbing,
arrays — and cap per category; template collapse is the standard failure); **bound the
ratio** to ~10–20% of total (loops are 0.07% of real requests; at 30% the prior shifts and
the model starts emitting loops for simple asks, violating the restraint the few-shots
enforce); **single-line forms only** (`stop_tokens()` includes `\n` and the grammar forbids
newlines, so multi-line fish is physically unemittable).

Uniform-random labeling and contrastive pairs are **two different mechanisms**. One ratio
must not govern both.

### Generation mechanics

Do not bulk-generate through Do itself — its daemon is built for interactive one-shot use
and llama.cpp's `/completion` is serial, so 50k greedy generations would run 4–7 hours and
best-of-*n* is out of reach. Import Do's `prompt.py` and apply the same GBNF under vLLM:
same distribution, batched, 50k×8 in well under an hour on the 4090. Use Do's daemon on a
few hundred samples to *prove* prompt equivalence, then generate in bulk.

### Known caveats

**The oracle can only adjudicate about two thirds of the corpus.** 35.3% of rows call a
primary program the sandbox does not have — `aws`, `tldr`, `az`, `gcloud`, `kubectl` and a
long tail of **2,699 distinct** tools, of which the top ten cover only 9%. Installing them
is not an option. Those rows return `NO_VERDICT`, not a match, so the pipeline must decide
explicitly what happens to them: drop them (losing a third of the data, skewed toward exotic
tooling), fall back to an LLM judge, or keep the unverified gold. Whichever is chosen, note
that execution verification silently *cannot* reach them.

The oracle is also imperfect where it does apply: ALFA gold is unverified, so a mismatch is
not proof the candidate is wrong. Track the reject pile rather than discarding it silently — a low-yield instruction
is usually either a hard case or a bad gold. Consider an LLM-judge as secondary arbiter on
mismatches, and decide explicitly whether unresolved instructions are dropped or fall back
to verified gold.

## 5. Evaluation

- **Baseline first.** Score the un-fine-tuned model on the harness *before* training, or the
  result is uninterpretable. Include **stock 1.5B** as the bar to beat (see Thesis).
- **Main metric — InterCode-ALFA style.** Execute candidate and reference, LLM-judge
  *output* equivalence, not command text. Accept either `bash` or `bash2` as reference.
  Executing the candidate in its labeled shell is what makes cross-shell scoring work at all.
- **Divergence suite.** ~100 hand-written items where the shells genuinely differ. Held out,
  never in training, and *not* produced by the teacher — otherwise it measures the teacher.
- **Leakage rate.** % of fish-labeled prompts emitting bash-only syntax. The aggregate is
  flattered by the agnostic majority; this is the number that says whether conditioning took.
- **Policy conformance.** Now that gold no longer matches Do's rules, track it directly:
  `;`-vs-`&&` chaining, unrequested flags. This is what distillation is supposed to buy.
- **Use Do's decoding path.** GBNF grammar, `stop_tokens()`, `extract()`, `temperature = 0.0`.
  Plain `generate()` measures a system we don't ship.
- **Sandbox.** Disposable container, seeded filesystem, per-command timeouts, no network. The
  corpus is unverified and contains destructive commands. The pipeline in §4 executes
  *everything*, so this is now load-bearing infrastructure, not just an eval concern.

## 6. Prerequisites

- **Sandbox — built.** [sandbox/](sandbox/) defines a Debian image with bash, zsh and fish
  plus the corpus's common tooling, and [scripts/shellrun.py](scripts/shellrun.py) executes
  commands in it and compares results (`Verdict.MATCH` / `DIFFER` / `NO_VERDICT`). Each
  command gets a fresh copy of a fixed seed tree, so mutating commands cannot contaminate
  the next one. Not yet built as an image: the Docker daemon is not running on this machine
  (`sudo systemctl enable --now docker`, then add yourself to the `docker` group).
- **Still to build:** the three-tier classifier (§4 step 3) and the eval harness (§5), both
  on top of `shellrun`.
- **Free win:** Do's grammar is ``first ::= [^`\n\r]`` / `rest ::= [^\n\r]`, so it only bans
  a *leading* backtick. Extending `rest` to ban backticks outright removes 2.47% of fish
  breakage at zero training cost — fish has no backtick substitution, and `$(...)` is correct
  in all three shells.

## 7. Success criteria & risks

Define before running: target ALFA accuracy, target leakage rate, and a **floor on bash
performance** we refuse to regress below.

- **Primary risk:** at 494M params, conditioning may buy fish accuracy at bash's expense.
  The bash floor is the tripwire.
- **Distillation risk:** rejection sampling biases toward instructions the teacher finds
  easy. If yield correlates with difficulty, the training set gets easier than reality —
  watch yield rate against ALFA's `difficulty` column on the test split.
- **Secondary risk:** over-training on noisy data (see §2, epochs).

## 8. Export

bf16 safetensors → GGUF (Q4_K_M, matching what Do ships); smoke-test under llama.cpp with
Do's ChatML tokens and grammar. Cheap to check early, annoying to discover at the end.
