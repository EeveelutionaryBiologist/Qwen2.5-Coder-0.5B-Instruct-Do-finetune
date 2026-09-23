# Qwen2.5-Coder-Do-finetune

Fine-tuning Qwen2.5-Coder-Instruct for [Do](https://github.com/EeveelutionaryBiologist/Do),
a natural language to shell translator.

Do works out which shell you're actually in and puts that in the prompt, so the model has
to handle fish and zsh, not just bash. It runs on the CPU via llama.cpp, which is the whole
reason for using a model this small.

Nothing has been trained yet. The training script works; the multi-shell data pipeline
doesn't exist (§4 of the plan).

Do currently ships stock 1.5B Q4_K_M. The point of all this is that a fine-tuned 0.5B
should beat it while decoding roughly 3x faster. 

## What's here

- `finetune-plan.md` — the plan and the reasoning behind it. Start here.
- `scripts/finetune_do.py` — training.
- `scripts/download_base_model.py`, `scripts/download_example_dataset.py` — standalone
  [uv](https://docs.astral.sh/uv/) scripts, nothing to install.
- `scripts/vendor/` — a copy of Do's prompt module, see below.
- `scripts/nl2sh_legacy/` — upstream NL2SH's training script, unmodified, kept for reference.

`model/`, `data/` and `runs/` are gitignored.

## Setup

Get the data (writes `data/example/{train,test}.csv`):

```sh
./scripts/download_example_dataset.py
```

Everything else wants a CUDA GPU. Built against a 24 GiB 4090.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
export HF_TOKEN=...   # optional, but the Hub throttles anonymous pulls
```

## Training

Check the data builds and the prompt looks right before burning an hour of GPU:

```sh
python scripts/finetune_do.py --dry-run
python scripts/finetune_do.py --max-steps 10
```

Then:

```sh
python scripts/finetune_do.py                                          # 0.5B
python scripts/finetune_do.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct # 1.5B
```

Weights come from the Hub unless you point `--model` at a local path. Output goes to
`runs/do-sft/`, final model in `runs/do-sft/final/`.

The 1.5B run only fits because `--optim auto` switches to 8-bit AdamW. Plain AdamW wants
23.0 GiB and you have about 23. Don't override it unless you've done the arithmetic.

`--epochs` defaults to 3. The plan suggests sweeping 2–4; upstream used 10, which is far
too many for data this noisy.

`--help` for the rest.

## The prompt

Training uses Do's actual prompt, few-shot examples and all, with loss computed only on
the command. Get this wrong and you train the model to recite `ls -a` forty thousand times.

`scripts/vendor/do_prompt.py` is a byte-identical copy of Do's `prompt.py`. Copying it is
ugly, but the Do checkout isn't guaranteed to exist on whatever machine you're training on.
The obvious failure mode is the copy going stale, so `finetune_do.py` hashes it against a
real Do checkout when it can find one and refuses to run if they differ
(`--allow-prompt-drift` if you really mean it). Every checkpoint gets a `do_prompt.json`
with the version and hash it was trained against.

Re-vendoring instructions are in `scripts/vendor/PROVENANCE.md`.

## Evaluation

Doesn't exist yet. When it does it'll execute commands and check they do the right thing in
the right shell, because string-matching against a reference command is meaningless here —
there are usually several correct answers. §5 of the plan.

Eval loss during training is a smoke signal, nothing more. Don't read anything into it.

## Credits

Dataset and the original training recipe are from
[NL2SH](https://github.com/westenfelder/NL2SH) and
[NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA) (MIT), out of
[this paper](https://arxiv.org/abs/2502.06858).
