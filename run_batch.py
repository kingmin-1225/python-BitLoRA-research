import os
import sys
import subprocess
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

adapters = ["fp32", "ternary", "binary"]
ranks = [4, 8, 16]

start_all = time.time()

for adapter in adapters:
    for r in ranks:
        print(f"\n{'='*50}")
        print(f"실행 중: Adapter={adapter}, Rank={r}")
        print(f"{'='*50}")
        
        cmd = [sys.executable, "new_train.py", "--adapter_type", adapter, "--r", str(r)]
        
        try:
            with open("all_experiments.log", "a") as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=f)
            print(f"SUCCESS: {adapter} - r{r}")
        except subprocess.CalledProcessError as e:
            print(f"FAIL: {adapter} - r{r} (ERROR: {e})")