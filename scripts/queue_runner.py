"""GPU job queue with lanes.

Jobs are lines ``name<TAB>command`` in queue/jobs.tsv (re-read after every job, so jobs can be
appended while the runner works). A job is skipped if queue/done/<name> or queue/failed/<name> exists.
Logs go to logs/jobs/<name>.log. The runner exits when no pending jobs of its lane are left and
queue/KEEPALIVE is absent, or immediately when queue/STOP exists.

Lanes. A runner serves one lane (``--lane``, default main) and holds that lane's exclusive flock while
a job runs, so GPU users that follow the same convention never overlap within a lane:
  main  -> GPU_LOCK        (default /root/workspace/data_raw/al_labeling/gpu.lock): trainings and heavy evals
  light -> GPU_LIGHT_LOCK  (default /root/workspace/data_raw/al_labeling/gpu_light.lock): evals whose
           measured peak GPU memory is <= 12 GB (scripts/gpu_peak.py), run next to a main-lane job.
A job's lane is given by a leading ``LANE=light`` token in its command (no token: main) and can be
overridden by queue/lane_override/<name> (file content = lane). A leading ``AFTER=a,b`` token defers the
job until queue/done/a and queue/done/b exist (cross-lane dependencies, e.g. an eval after its training).
If a light-lane job fails with a CUDA OOM, it is moved to the main lane (override file + a line in
queue/oom_events.log) instead of being marked failed.
"""
import argparse
import fcntl
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "queue"
DONE = Q / "done"
FAILED = Q / "failed"
OVERRIDE = Q / "lane_override"
LOGS = ROOT / "logs" / "jobs"
MIN_FREE = 50 * 2**30
LOCKS = {"main": os.environ.get("GPU_LOCK", "/root/workspace/data_raw/al_labeling/gpu.lock"),
         "light": os.environ.get("GPU_LIGHT_LOCK", "/root/workspace/data_raw/al_labeling/gpu_light.lock")}
TOKEN = re.compile(r"^\s*(LANE|AFTER)=(\S+)\s+")
for d in (DONE, FAILED, LOGS, OVERRIDE):
    d.mkdir(parents=True, exist_ok=True)


def ts():
    return time.strftime("%H:%M:%S")


def parse(cmd):
    """Strip leading LANE=/AFTER= tokens; return (lane, deps, cmd)."""
    lane, deps = "main", []
    while m := TOKEN.match(cmd):
        if m.group(1) == "LANE":
            lane = m.group(2)
        else:
            deps = [x for x in m.group(2).split(",") if x]
        cmd = cmd[m.end():]
    return lane, deps, cmd


def pending(lane):
    for line in (Q / "jobs.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, cmd = line.split("\t", 1)
        if (DONE / name).exists() or (FAILED / name).exists():
            continue
        jl, deps, cmd = parse(cmd)
        if (OVERRIDE / name).exists():
            jl = (OVERRIDE / name).read_text().strip()
        if jl != lane or not all((DONE / d).exists() for d in deps):
            continue
        return name, cmd
    return None


def is_oom(name):
    try:
        tail = (LOGS / f"{name}.log").read_text(errors="ignore")[-20000:]
    except OSError:
        return False
    return "CUDA out of memory" in tail or "OutOfMemoryError" in tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default=os.environ.get("QUEUE_LANE", "main"), choices=list(LOCKS))
    lane = ap.parse_args().lane
    lock_path = LOCKS[lane]
    print(f"[{ts()}] runner lane={lane} lock={lock_path} pid={os.getpid()}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    while not (Q / "STOP").exists():
        job = pending(lane)
        if job is None:
            if (Q / "KEEPALIVE").exists():
                time.sleep(60)
                continue
            break
        name, cmd = job
        while (free := shutil.disk_usage("/").free) < MIN_FREE:  # shared disk: never go below the agreed floor
            print(f"[{ts()}] waiting for disk space ({free / 2**30:.1f} GiB free)", flush=True)
            time.sleep(300)
        with open(lock_path, "a") as lock:
            tw = time.time()
            fcntl.flock(lock, fcntl.LOCK_EX)  # blocks while another GPU user holds this lane's lock
            if time.time() - tw > 5:
                print(f"[{ts()}] waited {(time.time() - tw) / 60:.1f} min for {lock_path}", flush=True)
            t0 = time.time()
            print(f"[{ts()}] start {name}", flush=True)
            with open(LOGS / f"{name}.log", "w") as log:
                rc = subprocess.call(cmd, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            fcntl.flock(lock, fcntl.LOCK_UN)
        minutes = (time.time() - t0) / 60
        if rc != 0 and lane == "light" and is_oom(name):
            (OVERRIDE / name).write_text("main\n")
            with open(Q / "oom_events.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name}\tOOM in light lane -> main\n")
            os.replace(LOGS / f"{name}.log", LOGS / f"{name}.oom_light.log")
            print(f"[{ts()}] OOM {name} in light lane -> moved to main lane ({minutes:.1f} min)", flush=True)
            continue
        target = DONE if rc == 0 else FAILED
        (target / name).write_text(f"rc={rc} minutes={minutes:.1f} lane={lane}\n")
        print(f"[{ts()}] {'done' if rc == 0 else 'FAILED'} {name} ({minutes:.1f} min)", flush=True)
    print("queue empty", flush=True)


if __name__ == "__main__":
    main()
