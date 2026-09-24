# Sandbox

Disposable environment for executing shell commands, used by the data pipeline
(`finetune-plan.md` §4) and the eval harness (§5).

Two reasons it exists:

1. **fish.** Classifying and verifying fish commands needs a fish to run them in.
2. **Safety.** The corpus is unverified and contains `rm`, `dd` and worse. It does
   not run on a host you care about.

## Build

```sh
docker build -t do-sandbox sandbox/
```

Needs a running Docker daemon. If `docker info` fails with a socket error:

```sh
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"   # then log out and back in
```

## Use

```python
from shellrun import DockerRunner

with DockerRunner() as run:
    print(run("fish", "for f in *.log; echo $f; end"))
```

Each command gets a fresh copy of the seed tree, so mutating commands can't
contaminate the next one. Containers run with `--network none`, a pids limit and
a memory cap.

## The seed tree

`seed.sh` builds a fixed tree: known filenames, contents, sizes and mtimes. Output
comparison is only meaningful if both sides acted on an identical starting state,
so nothing in it may depend on the host, the clock, or creation order.

`big.bin` is a 150M sparse file, so `find . -size +100M` has something to find
without costing 150M of disk.

## Caveats

- **`command not found` compares equal.** `aws`, `docker`, `npm`, `gcloud` and
  friends appear in the corpus and are not installed. Two commands that both fail
  identically will look equivalent. Callers must treat a failed *reference*
  execution as "no verdict", not as a match.
- fish here is whatever Debian bookworm ships. Check it against the version your
  users run before trusting fish-specific results.
