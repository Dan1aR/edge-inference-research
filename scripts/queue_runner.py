"""Sequential GPU job queue.

Jobs are lines ``name<TAB>command`` in queue/jobs.tsv (re-read after every job, so jobs can be
appended while the runner works). A job is skipped if queue/done/<name> exists. Logs go to
logs/jobs/<name>.log. The runner exits when no pending jobs are left and queue/KEEPALIVE is absent,
or immediately when queue/STOP exists.
Every job holds an exclusive flock on GPU_LOCK (env, default: the machine-wide lock /root/workspace/data_raw/al_labeling/gpu.lock) while it runs, so GPU users that
follow the same convention never overlap.
"""
import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "queue"
DONE = Q / "done"
FAILED = Q / "failed"
LOGS = ROOT / "logs" / "jobs"
MIN_FREE = 50 * 2**30
GPU_LOCK = os.environ.get("GPU_LOCK", "/root/workspace/data_raw/al_labeling/gpu.lock")  # machine-wide GPU lock
for d in (DONE, FAILED, LOGS):
    d.mkdir(parents=True, exist_ok=True)


def pending():
    for line in (Q / "jobs.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, cmd = line.split("\t", 1)
        if not (DONE / name).exists() and not (FAILED / name).exists():
            return name, cmd
    return None


def main():
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    while not (Q / "STOP").exists():
        job = pending()
        if job is None:
            if (Q / "KEEPALIVE").exists():
                time.sleep(60)
                continue
            break
        name, cmd = job
        while (free := shutil.disk_usage("/").free) < MIN_FREE:  # shared disk: never go below the agreed floor
            print(f"[{time.strftime('%H:%M:%S')}] waiting for disk space ({free / 2**30:.1f} GiB free)", flush=True)
            time.sleep(300)
        with open(GPU_LOCK, "a") as lock:
            tw = time.time()
            fcntl.flock(lock, fcntl.LOCK_EX)  # blocks while another GPU user holds the lock
            if time.time() - tw > 5:
                print(f"[{time.strftime('%H:%M:%S')}] waited {(time.time() - tw) / 60:.1f} min for {GPU_LOCK}", flush=True)
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] start {name}", flush=True)
            with open(LOGS / f"{name}.log", "w") as log:
                rc = subprocess.call(cmd, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            fcntl.flock(lock, fcntl.LOCK_UN)
        minutes = (time.time() - t0) / 60
        target = DONE if rc == 0 else FAILED
        (target / name).write_text(f"rc={rc} minutes={minutes:.1f}\n")
        print(f"[{time.strftime('%H:%M:%S')}] {'done' if rc == 0 else 'FAILED'} {name} ({minutes:.1f} min)", flush=True)
    print("queue empty", flush=True)


if __name__ == "__main__":
    main()
