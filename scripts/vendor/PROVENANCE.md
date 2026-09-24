# Vendored files

## `do_prompt.py`

A **byte-identical** copy of Do's prompt module. Vendored rather than imported
because the Do checkout is not guaranteed to exist on the training machine.

| | |
|---|---|
| Source | `Do/do/prompt.py` |
| Upstream commit | `d498c1b748c9fc445580cc1a29a803740ed25d89` (2026-09-16) |
| sha256 | `016eddd1c99dac444ea25bffed47b8cd9e116669da94049a34ccabfa12cdad0e` |
| `PROMPT_VERSION` | `1.0` |

Do not edit this copy. `finetune_do.py` compares it against the upstream file
whenever a Do checkout is reachable, and refuses to train on a stale prompt --
a checkpoint is only valid for the prompt it was trained against.

To re-vendor after an upstream prompt change:

```sh
cp ../Do/do/prompt.py scripts/vendor/do_prompt.py
sha256sum scripts/vendor/do_prompt.py   # update the table above
```

Bump `PROMPT_VERSION` upstream when the prompt changes semantically, so existing
checkpoints can be told apart from ones trained on the new prompt.
