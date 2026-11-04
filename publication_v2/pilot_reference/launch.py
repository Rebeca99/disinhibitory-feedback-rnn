"""Queue one training job per allocated GPU, or one CPU smoke-test worker."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--gpus", nargs="+", help="CUDA device IDs within your allocation; CPU is a smoke-test option")
    p.add_argument("--indices", nargs="+", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    plan_path = Path(args.plan).resolve()
    jobs = json.loads(plan_path.read_text())["jobs"]
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = args.gpus or (allocated.split(",") if allocated else None)
    if not gpus:
        p.error("Supply --gpus using allocated GPUs, or run inside a CUDA_VISIBLE_DEVICES allocation")
    if allocated and gpus != ["cpu"] and not set(gpus).issubset(set(allocated.split(","))):
        p.error("Requested GPUs are outside CUDA_VISIBLE_DEVICES; use its allocation IDs, not renumbered IDs")
    indices = args.indices if args.indices is not None else list(range(len(jobs)))
    from core import code_digest
    current_code = code_digest()
    for i in indices:
        metadata = Path(jobs[i]["output"]) / "job.json"
        if metadata.exists() and json.loads(metadata.read_text()).get("code_sha256") != current_code:
            raise SystemExit("Code changed since run {}. Use a new output root; do not mix versions.".format(i))
    pending = deque(i for i in indices if not (Path(jobs[i]["output"]) / "final.pt").exists())
    print("Pending jobs: {} on devices {}".format(len(pending), ", ".join(gpus)), flush=True)
    if args.dry_run:
        for i in pending:
            print(i, jobs[i]["id"])
        return
    active, failures = {}, []
    stopped = [False]
    def stop(*unused):
        stopped[0] = True
        for process, logfile, _ in active.values():
            process.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    script = Path(__file__).resolve().parent / "train_publication.py"
    while active or (pending and not stopped[0]):
        for gpu in gpus:
            if gpu not in active and pending and not stopped[0]:
                index = pending.popleft()
                output = Path(jobs[index]["output"])
                output.mkdir(parents=True, exist_ok=True)
                logfile = open(output / "worker.log", "a")
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                if gpu != "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                command = [sys.executable, str(script), "--plan", str(plan_path), "--job-index", str(index),
                           "--device", "cpu" if gpu == "cpu" else "cuda:0"]
                process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, env=env)
                active[gpu] = (process, logfile, index)
                print("START", gpu, index, jobs[index]["id"], flush=True)
        for gpu, (process, logfile, index) in list(active.items()):
            code = process.poll()
            if code is not None:
                logfile.close()
                if code != 0 or not (Path(jobs[index]["output"]) / "final.pt").exists():
                    failures.append(index)
                print("END", gpu, index, "exit", code, flush=True)
                del active[gpu]
        if active:
            time.sleep(2)
    if failures:
        print("Incomplete/failed indices:", failures)
        print("Inspect worker.log, then rerun the same launch command to resume.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
