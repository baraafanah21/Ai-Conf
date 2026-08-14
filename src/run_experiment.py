"""Run the full experiment on ONE system: N repetitions of run_trial.py.

Each trial runs in a fresh Python subprocess so that memory state, allocator
caches, and JIT state never leak between repetitions (each of the 50 runs is
an independent sample, as in Rahman et al. 2024).

Usage (on each machine/OS, with the venv active):
    python src/run_experiment.py --os-label ubuntu
    python src/run_experiment.py --os-label win11_machine1
    python src/run_experiment.py --os-label fedora
    python src/run_experiment.py --os-label win11_machine2

    python src/run_experiment.py --os-label ubuntu --dry-run   # 3 quick runs

Results are appended to results/raw/<file>.csv (one row per trial).
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

OS_LABELS = {
    "ubuntu": "ubuntu_raw.csv",
    "win11_machine1": "win11_m1_raw.csv",
    "fedora": "fedora_raw.csv",
    "win11_machine2": "win11_m2_raw.csv",
}
RESULT_PREFIX = "RESULT_JSON: "


def run_one_trial(trial_script: Path, args: argparse.Namespace, run_idx: int) -> dict:
    cmd = [
        sys.executable, str(trial_script),
        "--os-label", args.os_label,
        "--data-dir", args.data_dir,
        "--run-idx", str(run_idx),
        "--batch-size", str(args.batch_size),
        "--threads", str(args.threads),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"trial {run_idx} failed (exit {proc.returncode})")
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
    raise RuntimeError(f"trial {run_idx}: no RESULT_JSON line in output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os-label", required=True, choices=sorted(OS_LABELS))
    parser.add_argument("--runs", type=int, default=50,
                        help="repetitions (50 = same as the reference paper)")
    parser.add_argument("--dry-run", action="store_true",
                        help="quick check: run only 3 repetitions")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cooldown", type=float, default=2.0,
                        help="seconds to sleep between trials (system settle)")
    args = parser.parse_args()

    n_runs = 3 if args.dry_run else args.runs
    trial_script = Path(__file__).parent / "run_trial.py"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OS_LABELS[args.os_label]

    data_dir = Path(args.data_dir)
    for required in ("model_weights.pt", "test_set.pt"):
        if not (data_dir / required).exists():
            sys.exit(
                f"ERROR: {data_dir / required} not found. Run model_setup.py once on the "
                "reference machine and copy the data/ directory here first."
            )

    print(f"os_label={args.os_label}  runs={n_runs}  output={out_path}")
    writer = None
    t_start = time.perf_counter()
    with open(out_path, "a", newline="") as f:
        for i in range(n_runs):
            result = run_one_trial(trial_script, args, i)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                if f.tell() == 0:
                    writer.writeheader()
            writer.writerow(result)
            f.flush()
            print(f"run {i + 1:>3}/{n_runs}  "
                  f"acc={result['accuracy']:.4f}  "
                  f"time={result['total_time_s']:.2f}s  "
                  f"mem={result['peak_rss_mb']:.0f}MB  "
                  f"cpu={result['cpu_percent']:.0f}%")
            if i < n_runs - 1 and args.cooldown > 0:
                time.sleep(args.cooldown)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {n_runs} runs in {elapsed / 60:.1f} min -> {out_path}")
    if args.dry_run:
        print("This was a --dry-run. Re-run without --dry-run for the full 50 repetitions.")


if __name__ == "__main__":
    main()
