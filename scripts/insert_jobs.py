"""Insert job lines (stdin) into queue/jobs.tsv before the first line whose name starts with --before."""
import sys

before = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--before" else None
new = [l for l in sys.stdin.read().splitlines() if l.strip()]
lines = open("queue/jobs.tsv").read().splitlines()
idx = next((i for i, l in enumerate(lines) if before and l.startswith(before)), len(lines))
open("queue/jobs.tsv", "w").write("\n".join(lines[:idx] + new + lines[idx:]) + "\n")
print(f"inserted {len(new)} jobs at line {idx}")
