# Fine-tuning plan

Fine-tune `Qwen2.5-Coder-0.5B-Instruct` for the [Do](../Do) NL→shell translator,
conditioned on the target shell (**bash / zsh / fish**), which Do auto-detects and
injects into the prompt.

- **Training target:** single RTX 4090 (24 GB).
- **Delivery format:** GGUF, for Do's llama.cpp backend.
- **Reference:** upstream [NL2SH](https://github.com/westenfelder/NL2SH) shipped a
  `Qwen2.5-Coder-0.5B-Instruct-NL2SH`; their training script is vendored verbatim at
  [scripts/nl2sh_legacy/finetune.py](scripts/nl2sh_legacy/finetune.py).

---

## 1. Regime — full fine-tune via TRL `SFTTrainer`

**Full FT, not LoRA/QLoRA.** At 494M params, AdamW mixed-precision costs ~9 GB:

| Component | VRAM |
|---|---|
| fp32 master weights | 2.0 GB |
| Adam `m` + `v` (fp32) | 4.0 GB |
| bf16 weights + grads | 2.0 GB |
| Activations @ 150 tok | < 1 GB |

~15 GB headroom on a 4090. PEFT solves a problem we don't have, converges worse for a
real domain shift, and complicates checkpoint export.

**Stack:** `torch`, `transformers` (v5), `trl==1.13.*` (v1.13.0, 2026-09-10), `datasets`,
`accelerate`, optional `wandb`. Pin the TRL minor — they are actively pruning old APIs.

**Starting hyperparameters**, from the legacy script: `lr=1e-5`, effective batch 75,
`max_len=150`, bf16. Note upstream used those for the **3B** model.

**Three fixes over the legacy script:**
1. Completion-only loss — it masks only pad tokens, so it trains on the system prompt and
   the English instruction too.
2. Dynamic padding or packing, not `padding="max_length"` — most pairs are far under 150.
3. Fewer epochs — 10 epochs over 40k *unverified* pairs invites memorizing noise. Sweep 2–4.

## 2. Prompt — mirror Do exactly

Import `SYSTEM`, `FEW_SHOTS` and `build()` from [../Do/do/prompt.py](../Do/do/prompt.py)
at data-build time. **Do not copy-paste** — rule 4 already changed once (made
shell-conditional), and a stale copy silently invalidates every rendered example. Stamp
`PROMPT_VERSION` into the run config and model card; a prompt change invalidates the
fine-tune.

> **Trap:** use TRL's `{"prompt", "completion"}` dataset format, **not**
> `assistant_only_loss=True`. Do's prompt carries 8 few-shot assistant turns; assistant-only
> masking trains on all nine, memorizing `ls -a` / `ps aux` 40,639 times.

`build()` makes `shell:`, `os:`, `cwd:` and `files here:` all optional — 16 possible shapes.
Sample those combinations to match Do's runtime distribution, or every real invocation with
a cwd and file listing is out-of-distribution.

## 3. Data

Measured over `data/example` (NL2SH-ALFA):

| Finding | Value |
|---|---|
| Train rows | 40,639 (`nl`,`bash`, unverified) |
| Test rows | 300 rows × 2 refs (`bash`/`bash2`) = the 600 "pairs" on the card |
| Parses identically under bash **and** zsh | 99.8% (1,500 sampled, 0 zsh-only failures) |
| Contains any fish-divergent construct | 3.51% — backticks alone are 2.47 pts of that |
| Genuine structural divergence | ~1% (`VAR=x cmd` 1.07%, `do/done` 0.07%, `then/fi` 0.03%) |
| Test `nl` also appearing in train | 3 (1.0%) — dedupe |
| Duplicate `nl` within train | 729 distinct / 2,480 rows — dedupe before labeling |

**zsh is nearly free; fish is the entire job; and the divergent slice is far too small for
random sampling to ever teach it.** If ~96% of examples have a gold command that is identical
regardless of shell, the model minimizes loss perfectly by ignoring the `shell:` line — then
emits bash when asked for fish, while the aggregate score still looks fine.

### Pipeline

1. **Three-tier classification.** (i) structurally impossible in fish → must translate;
   (ii) provably identical across shells → agnostic; (iii) parses everywhere but *behaves*
   differently — unquoted `$var` word splitting, `**` recursion, glob-no-match (bash passes
   the literal, fish errors). Tier iii needs execution, not regex.
2. **Agnostic majority (tier ii).** Assign bash/zsh/fish uniformly at random, **one label per
   row**. This stops "fish" becoming a rare token correlated with odd output. Do *not* emit
   three copies of `ls -a` — that triples compute and actively reinforces label-ignoring.
3. **Divergent slice (tiers i/iii).** Deliberate **contrastive pairs**: same `nl`, different
   shell labels, genuinely different gold commands. Highest-leverage item in this document —
   the model cannot minimize loss without attending to the shell line.
4. **Synthetic fish inflation** — worth doing, under four conditions:
   - **Verify by execution.** `fish -n` plus sandboxed output-equivalence against the bash
     original. A 0.5B student memorizes generator hallucinations rather than averaging them out.
   - **Breadth over depth.** Enumerate a divergence taxonomy (one-line loops, conditionals,
     `set -x` env vars, local vars, `math`, `$status`, prefix assignment, globbing, arrays) and
     cap per category. Template collapse is the standard failure.
   - **Bound the ratio.** Loops are 0.07% of real requests; if they become 30% of fish data the
     prior shifts and the model starts emitting loops for simple asks — violating exactly the
     restraint the few-shots exist to enforce. Keep the divergent slice ~10–20% of total.
   - **Single-line forms only.** `stop_tokens()` includes `\n` and the grammar forbids newlines,
     so multi-line fish is physically unemittable.

Uniform-random labeling and contrastive pairs are **two different mechanisms**. One ratio
must not govern both.

## 4. Evaluation

- **Baseline first.** Score the un-fine-tuned model on the harness *before* training, or the
  result is uninterpretable. Optionally a bash-only control run, to separate "SFT helped"
  from "conditioning helped".
- **Main metric — InterCode-ALFA style.** Execute candidate and reference, LLM-judge
  *output* equivalence, not command text. Accept either `bash` or `bash2` as reference.
  Executing the candidate in its labeled shell is what makes cross-shell scoring work at all.
- **Divergence suite.** ~100 hand-written items where the shells genuinely differ. Held out,
  never in training, and *not* produced by the training-data generator — otherwise it measures
  the generator.
- **Leakage rate.** % of fish-labeled prompts emitting bash-only syntax. The aggregate is
  flattered by the agnostic majority; this is the number that says whether conditioning took.
- **Use Do's decoding path.** GBNF grammar, `stop_tokens()`, `extract()`. Plain `generate()`
  measures a system we don't ship.
- **Sandbox.** Disposable container, seeded filesystem, per-command timeouts, no network. The
  corpus is unverified and contains destructive commands.

## 5. Prerequisites

- **fish is not installed here** (bash and zsh are). Steps 3 and 4 need a container with all
  three shells. This is blocking and currently the least-specified piece.
- **Free win:** Do's grammar is ``first ::= [^`\n\r]`` / `rest ::= [^\n\r]`, so it only bans a
  *leading* backtick. Extending `rest` to ban backticks outright removes 2.47% of fish
  breakage at zero training cost — fish has no backtick substitution, and `$(...)` is correct
  in all three shells.

## 6. Success criteria & risks

Define before running: target ALFA accuracy, target leakage rate, and a **floor on bash
performance** we refuse to regress below.

- **Primary risk:** at 494M params, conditioning may buy fish accuracy at bash's expense.
  The bash floor is the tripwire.
- **Secondary risk:** over-training on unverified data (see §1, epochs).

## 7. Export

bf16 safetensors → GGUF; smoke-test under llama.cpp with Do's ChatML tokens and grammar.
