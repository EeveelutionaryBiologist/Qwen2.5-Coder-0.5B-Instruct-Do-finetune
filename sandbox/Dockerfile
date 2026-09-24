# Sandbox for executing corpus and candidate commands (finetune-plan.md §4, §5).
#
# The NL2SH-ALFA corpus is unverified and contains destructive commands, so
# nothing here should ever run on a host you care about. Containers are started
# with --network none and a read-only pristine seed at /opt/seed.
FROM debian:bookworm-slim

# The three shells Do targets. Versions are whatever bookworm ships; check them
# against what your users actually run before trusting fish results.
# Everything else is chosen from the head-command distribution of the corpus:
# find/xargs/sort/head/shuf/awk/grep/sed/cut/tr/date account for most of it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash zsh fish \
        coreutils findutils grep sed gawk diffutils \
        procps util-linux bsdmainutils \
        tar gzip xz-utils \
        file less git rsync jq bc \
    && rm -rf /var/lib/apt/lists/*

# Pristine tree. Each command gets a fresh copy of this, never the original.
COPY seed.sh /opt/seed.sh
RUN /opt/seed.sh /opt/seed && chmod -R a-w /opt/seed

RUN useradd --create-home --shell /bin/bash runner
USER runner
WORKDIR /work

# Idle by default; the runner drives it with `docker exec`.
CMD ["sleep", "infinity"]
