"""Run src/eval.py for the base model and every adapter checkpoint, sequentially.

Finished runs (eval_results/<run>.jsonl exists) are skipped, so the script can be restarted after an interruption.

    python run_eval_batch.py --adapter_root to_eval --checkpoint checkpoint-1800
"""
import argparse
import os
import subprocess
import sys
import time

ADAPTERS = ["fp32", "ternary", "binary"]
RANKS = [4, 8, 16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_root", default="to_eval", help="contains <type>-r<r>/<checkpoint>/")
    parser.add_argument("--checkpoint", default="checkpoint-1800")
    parser.add_argument("--batch_size", type=int, default=8, help="keep fixed across runs: results depend on padding")
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--log", default="eval_results/eval_batch.log")
    args = parser.parse_args()

    runs = [("base", None, "base")]
    for adapter in ADAPTERS:
        for r in RANKS:
            path = os.path.join(args.adapter_root, f"{adapter}-r{r}", args.checkpoint)
            runs.append((adapter, path, f"{adapter}-r{r}-{args.checkpoint}"))

    os.makedirs(args.output_dir, exist_ok=True)
    start_all = time.time()
    for adapter_type, path, run_name in runs:
        if os.path.exists(os.path.join(args.output_dir, f"{run_name}.jsonl")):
            print(f"SKIP (done): {run_name}", flush=True)
            continue
        if path and not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
            print(f"SKIP (missing adapter): {path}", flush=True)
            continue

        cmd = [sys.executable, "src/eval.py", "--adapter_type", adapter_type,
               "--batch_size", str(args.batch_size), "--output_dir", args.output_dir]
        if path:
            cmd += ["--adapter_path", path]

        print(f"START: {run_name}", flush=True)
        start = time.time()
        with open(args.log, "a", encoding="utf-8") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        status = "SUCCESS" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
        print(f"{status}: {run_name} ({(time.time() - start) / 60:.1f} min)", flush=True)

    print(f"ALL DONE in {(time.time() - start_all) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
