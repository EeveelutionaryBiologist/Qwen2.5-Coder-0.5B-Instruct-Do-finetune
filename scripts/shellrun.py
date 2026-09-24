#!/usr/bin/env python3
"""Execute shell commands in a disposable sandbox and compare what they do.

Used by the data pipeline (finetune-plan.md §4), which keeps a candidate command
only when it produces the same result as the reference, and by the eval harness
(§5), which scores a model the same way.

Commands are never compared as strings. Two commands are equivalent when running
them against an identical filesystem produces the same output and exit status.

The corpus is unverified and contains destructive commands. DockerRunner is the
only backend that is safe for it; LocalRunner exists for developing on bash/zsh
and refuses to start unless you say you meant it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

SHELLS = ("bash", "zsh", "fish")
SEED_SH = Path(__file__).resolve().parent.parent / "sandbox" / "seed.sh"
DEFAULT_IMAGE = "do-sandbox"

# Every shell here reports "command not found" as 127.
NOT_FOUND = 127

# Run the command in a throwaway copy of the seed tree. The command itself
# arrives in $DO_CMD so it never has to survive another round of shell quoting.
_SCRIPT = r"""
set -u
d=$(mktemp -d) || exit 70
cp -a "$DO_SEED"/. "$d" || exit 70
cd "$d" || exit 70
timeout -k 1 "$DO_TIMEOUT" "$DO_SHELL" -c "$DO_CMD"
rc=$?
cd / || exit 70
chmod -R u+w "$d" 2>/dev/null
rm -rf "$d"
exit $rc
"""


@dataclass(frozen=True)
class Result:
    shell: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def not_found(self) -> bool:
        """The shell could not find the program at all (aws, docker, npm, ...).

        Distinct from a command that ran and failed: a not-found result carries
        no information about what the command would have done.
        """
        return self.returncode == NOT_FOUND

    @property
    def usable(self) -> bool:
        """Whether this execution says anything about the command's behaviour."""
        return not (self.timed_out or self.not_found)

    def __str__(self) -> str:
        tag = "timeout" if self.timed_out else f"rc={self.returncode}"
        return f"[{self.shell} {tag}] {self.stdout.strip()[:200]}"


class Verdict(Enum):
    MATCH = "match"
    DIFFER = "differ"
    NO_VERDICT = "no_verdict"


def normalize(text: str, *, sort_lines: bool = False) -> str:
    """Canonical form for comparison: no trailing blanks, optional line order."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    if sort_lines:
        lines.sort()
    return "\n".join(lines)


def compare(reference: Result, candidate: Result, *, sort_lines: bool = False) -> Verdict:
    """Did the candidate do the same thing as the reference?

    NO_VERDICT when the reference itself tells us nothing -- it timed out, or the
    program isn't installed. Without this, two commands that both fail with
    "command not found" would score as equivalent, and roughly 3% of the corpus
    calls tools the sandbox does not have (aws, docker, npm, gcloud).
    """
    if not reference.usable or candidate.timed_out:
        return Verdict.NO_VERDICT
    if reference.returncode != candidate.returncode:
        return Verdict.DIFFER
    same = normalize(reference.stdout, sort_lines=sort_lines) == normalize(
        candidate.stdout, sort_lines=sort_lines
    )
    return Verdict.MATCH if same else Verdict.DIFFER


class LocalRunner:
    """Runs on this machine. No isolation whatsoever.

    Only for developing against bash and zsh with commands you wrote yourself.
    Never point it at the corpus.
    """

    def __init__(self, timeout: int = 10, *, i_know_this_is_unsafe: bool = False):
        if not i_know_this_is_unsafe:
            raise RuntimeError(
                "LocalRunner executes commands directly on this machine. "
                "Use DockerRunner for anything from the corpus. Pass "
                "i_know_this_is_unsafe=True if you really mean it."
            )
        self.timeout = timeout
        self._seed = Path(tempfile.mkdtemp(prefix="do-seed-"))
        subprocess.run([str(SEED_SH), str(self._seed)], check=True)

    def __enter__(self) -> "LocalRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        shutil.rmtree(self._seed, ignore_errors=True)

    def __call__(self, shell: str, command: str) -> Result:
        if not shutil.which(shell):
            return Result(shell, command, NOT_FOUND, "", f"{shell}: not installed")
        env = {
            **os.environ,
            "DO_CMD": command,
            "DO_SHELL": shell,
            "DO_SEED": str(self._seed),
            "DO_TIMEOUT": str(self.timeout),
        }
        try:
            proc = subprocess.run(
                ["sh", "-c", _SCRIPT], env=env, capture_output=True,
                text=True, timeout=self.timeout + 5,
            )
        except subprocess.TimeoutExpired:
            return Result(shell, command, 124, "", "", timed_out=True)
        return Result(shell, command, proc.returncode, proc.stdout, proc.stderr,
                      timed_out=proc.returncode == 124)


class DockerRunner:
    """One long-lived container, one `docker exec` per command.

    A fresh container per command would be correct too, but at ~200ms of startup
    each that is 18 hours for a best-of-8 pass over the corpus. Isolation comes
    from the container; a clean filesystem comes from copying the seed per run.
    """

    def __init__(self, image: str = DEFAULT_IMAGE, timeout: int = 10,
                 memory: str = "512m", pids: int = 256):
        self.image, self.timeout = image, timeout
        self.name = f"do-sandbox-{uuid.uuid4().hex[:12]}"
        subprocess.run(
            ["docker", "run", "--rm", "--detach", "--name", self.name,
             "--network", "none", "--memory", memory, "--pids-limit", str(pids),
             self.image],
            check=True, capture_output=True, text=True,
        )

    def __enter__(self) -> "DockerRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        subprocess.run(["docker", "kill", self.name],
                       capture_output=True, check=False)

    def __call__(self, shell: str, command: str) -> Result:
        try:
            proc = subprocess.run(
                ["docker", "exec",
                 "-e", f"DO_CMD={command}", "-e", f"DO_SHELL={shell}",
                 "-e", "DO_SEED=/opt/seed", "-e", f"DO_TIMEOUT={self.timeout}",
                 self.name, "sh", "-c", _SCRIPT],
                capture_output=True, text=True, timeout=self.timeout + 15,
            )
        except subprocess.TimeoutExpired:
            return Result(shell, command, 124, "", "", timed_out=True)
        return Result(shell, command, proc.returncode, proc.stdout, proc.stderr,
                      timed_out=proc.returncode == 124)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run one command in every shell.")
    ap.add_argument("command")
    ap.add_argument("--local", action="store_true", help="unsafe: run on this machine")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    args = ap.parse_args()

    runner = (LocalRunner(i_know_this_is_unsafe=True) if args.local
              else DockerRunner(args.image))
    with runner as run:
        ref = run("bash", args.command)
        for sh in SHELLS:
            res = run(sh, args.command)
            note = "" if sh == "bash" else f"  -> {compare(ref, res).value}"
            print(f"{res}{note}")
