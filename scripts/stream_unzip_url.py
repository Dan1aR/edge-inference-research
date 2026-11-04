"""Download a zip over HTTP and extract it on the fly (no temporary zip on disk)."""
import sys
from pathlib import Path

import httpx
from stream_unzip import stream_unzip

url, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)


def zipped_chunks():
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        yield from r.iter_bytes(chunk_size=1 << 20)


n = 0
for name, _size, chunks in stream_unzip(zipped_chunks()):
    path = out_dir / name.decode()
    if name.endswith(b"/"):
        path.mkdir(parents=True, exist_ok=True)
        for _ in chunks:
            pass
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)
    n += 1
    if n % 10000 == 0:
        print(f"{n} files", flush=True)
print(f"done: {n} files -> {out_dir}", flush=True)
