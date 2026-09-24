#!/bin/sh
# Build a deterministic scratch filesystem at $1.
#
# Commands are compared by what they *do*, so both sides have to act on an
# identical tree. Everything here is fixed: names, contents, sizes, mtimes.
# Nothing may depend on the host, the clock, or the order files were created.
set -eu

target="${1:?usage: seed.sh DIR}"
rm -rf "$target"
mkdir -p "$target"
cd "$target"

# A stable mtime for everything, so `ls -l`, `find -newer` and friends agree.
STAMP=202601010000.00

mkdir -p src data logs .config

printf 'import sys\n\n\ndef main():\n    print("hello")\n    return 0\n\n\nif __name__ == "__main__":\n    sys.exit(main())\n' > main.py
printf 'alpha\nbeta\ngamma\ndelta\n' > notes.txt
printf '{\n  "name": "demo",\n  "version": "1.0"\n}\n' > config.json
printf 'secret\n' > .hidden

printf 'def helper():\n    return 1\n' > src/helper.py
printf 'def util():\n    return 2\n' > src/util.py
printf 'name,count\nfoo,3\nbar,7\nbaz,11\n' > data/report.csv

printf 'INFO start\nINFO ready\n' > logs/app.log
printf 'ERROR disk full\n' > logs/error.log
printf 'DEBUG tick\n' > debug.log
printf 'INFO boot\n' > app.log

# Sparse, so `find . -size +100M` has something to find without costing 150M.
truncate -s 150M big.bin

find . -exec touch -t "$STAMP" {} +
