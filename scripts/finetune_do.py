#!/usr/bin/env python3
"""Full fine-tune of Qwen2.5-Coder-Instruct for Do, via TRL SFTTrainer.

Adapted from scripts/nl2sh_legacy/finetune.py with the fixes from
finetune-plan.md §2, trained against Do's own prompt per §3:

  1. Completion-only loss. The legacy script masks only pad tokens, so it trains
     on the system prompt and the English instruction as well as the command.
     Here each example is a prompt-completion pair, so TRL masks the whole
     prompt -- including Do's eight few-shot turns -- and computes loss on the
     command alone.
  2. Dynamic padding (or packing), not padding="max_length".
  3. Fewer epochs -- 10 over 40k unverified pairs memorizes noise. Default 3,
     sweep 2-4.

The prompt is imported from ../Do/do/prompt.py rather than copied, so the
training prompt cannot drift from the inference prompt. PROMPT_VERSION is
recorded alongside the checkpoint: a prompt change invalidates the fine-tune.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DO_REPO = REPO_ROOT.parent / "Do"

# 8-bit optimizer states are what make the 1.5B run fit: standard AdamW needs
# 23.0 GiB against ~23 GiB usable, 8-bit brings it to 14.4 GiB (§1).
BNB_8BIT = "adamw_bnb_8bit"
TORCH_FUSED = "adamw_torch_fused"


def load_do_prompt(do_repo: Path):
    """Import Do's prompt module. Imported, never copied -- see §3."""
    if not (do_repo / "do" / "prompt.py").exists():
        raise SystemExit(f"Do's prompt module not found under {do_repo} -- pass --do-repo.")
    sys.path.insert(0, str(do_repo))
    from do import prompt as prompt_mod  # noqa: E402

    return prompt_mod


def render_prompt(P, request: str, shell: str, os_name: str, few_shots: bool) -> str:
    """Do's ChatML prompt for one request, ending at the assistant turn.

    With few_shots=True this is exactly `P.build(...)`. With few_shots=False we
    reassemble the same turn format without the eight restraint examples, which
    is only correct if Do drops them too -- see --no-few-shots.
    """
    if few_shots:
        return P.build(request, shell=shell, os_name=os_name)
    # P._turn is Do-internal; used so the turn format still has one definition.
    context = [f"shell: {shell}"] if shell else []
    if os_name:
        context.append(f"os: {os_name}")
    user = "\n".join(context + [request])
    return P._turn("system", P.SYSTEM) + P._turn("user", user) + f"{P.IM_START}assistant\n"


def build_formatter(P, args):
    """Row -> {"prompt", "completion"}. The completion stays bare: TRL appends
    the tokenizer's eos_token (<|im_end|> for Qwen Instruct) itself."""

    def to_prompt_completion(row: dict, index: int) -> dict:
        # §4 will ship a per-row shell label; until then fall back to --shell.
        shell = row.get("shell") or args.shell
        os_name = args.os_name
        # Do makes every context line optional, so the prompt has several
        # possible shapes at inference. Drop lines sometimes so a missing one is
        # not out-of-distribution. Seeded on the row index for reproducibility.
        local = random.Random(args.seed + index)
        if local.random() < args.context_dropout:
            shell = ""
        if local.random() < args.context_dropout:
            os_name = ""
        return {
            "prompt": render_prompt(P, row["nl"], shell, os_name, not args.no_few_shots),
            "completion": row["bash"],
        }

    return to_prompt_completion


def load_split(data_dir: Path, name: str, formatter):
    """Load one local CSV written by scripts/download_example_dataset.py."""
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/download_example_dataset.py first.")
    split = load_dataset("csv", data_files=str(path), split="train")
    # test.csv also carries bash2 and difficulty; drop everything but the pair.
    return split.map(formatter, with_indices=True, remove_columns=split.column_names)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                   help="model id or local path (default: %(default)s)")
    p.add_argument("--do-repo", type=Path, default=DEFAULT_DO_REPO,
                   help="checkout of the Do project to import the prompt from")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "example")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "do-sft")
    # --- prompt ---
    p.add_argument("--shell", default="bash",
                   help="shell label when the data has no `shell` column (§4 adds one)")
    p.add_argument("--os-name", default="Linux")
    p.add_argument("--context-dropout", type=float, default=0.1,
                   help="per-line chance of omitting a context line, since Do makes them optional")
    p.add_argument("--no-few-shots", action="store_true",
                   help="train without Do's few-shot block -- only correct if Do drops it too")
    # --- optimization ---
    # Fix 3: 3 epochs, not 10. Sweep 2-4.
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    # 15 x 5 = the legacy effective batch of 75.
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--grad-accum", type=int, default=5)
    # Do's prompt is ~400 tokens with few-shots, so the legacy cap of 150 would
    # truncate the answer off the end (TRL truncates keep_start).
    p.add_argument("--max-length", type=int, default=512)
    # Fix 2: dynamic padding is the default. Packing is faster still, but its
    # bfd strategy turns on padding-free, which needs FlashAttention 2/3 --
    # so it stays opt-in rather than surprising you on a fresh box.
    p.add_argument("--packing", action="store_true",
                   help="pack sequences into fixed-length blocks (needs flash-attn)")
    p.add_argument("--optim", default="auto", choices=["auto", BNB_8BIT, TORCH_FUSED],
                   help="auto: 8-bit AdamW for 1.5B and larger, fused AdamW otherwise")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="off by default: at these lengths activations are not the bottleneck")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--report-to", default="none", help='e.g. "wandb"')
    return p.parse_args()


def resolve_optim(choice: str, model_id: str) -> str:
    if choice != "auto":
        return choice
    small = any(tag in model_id for tag in ("0.5B", "0.5b"))
    return TORCH_FUSED if small else BNB_8BIT


def main() -> None:
    args = parse_args()
    optim = resolve_optim(args.optim, args.model)
    P = load_do_prompt(args.do_repo)
    formatter = build_formatter(P, args)

    train_dataset = load_split(args.data_dir, "train", formatter)
    # NOTE: this is the InterCode-ALFA test set. Eval loss here is a smoke signal
    # only -- the metric that decides anything is the execution harness in §5,
    # and selecting checkpoints on this split would contaminate that benchmark.
    eval_dataset = load_split(args.data_dir, "test", formatter)

    print(f"model={args.model} optim={optim} prompt_version={P.PROMPT_VERSION} "
          f"few_shots={not args.no_few_shots} "
          f"train={len(train_dataset)} eval={len(eval_dataset)}")
    print("--- example prompt ---")
    print(train_dataset[0]["prompt"][-220:])
    print(f"--- completion: {train_dataset[0]['completion']!r}")

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        # --- data ---
        max_length=args.max_length,
        packing=args.packing,
        shuffle_dataset=True,
        # --- loss (fix 1) ---
        # Stated rather than left to the default, so that a change to the dataset
        # shape fails loudly instead of silently training on the few-shot turns.
        completion_only_loss=True,
        # --- optimization ---
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim=optim,
        weight_decay=0.01,
        max_grad_norm=2.0,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        # --- bookkeeping ---
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=2000,
        save_total_limit=3,
        seed=args.seed,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    final = args.output_dir / "final"
    trainer.save_model(str(final))
    # A checkpoint is only valid for the prompt it was trained against.
    (final / "do_prompt.json").write_text(json.dumps({
        "prompt_version": P.PROMPT_VERSION,
        "few_shots": not args.no_few_shots,
        "shell": args.shell,
        "os_name": args.os_name,
        "context_dropout": args.context_dropout,
        "base_model": args.model,
    }, indent=2) + "\n")
    print(f"saved to {final}")


if __name__ == "__main__":
    main()
