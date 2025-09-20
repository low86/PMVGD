"""
Run the full pipeline in sequence: Phase1 -> Phase2 -> Phase3.
This script simply calls the three experiment launchers in order.
Author: Tian Yuxuan
Date: 2025-08-021
"""

import subprocess
import sys

if __name__ == '__main__':
    # You can simply call each script in turn. Using subprocess ensures each script runs
    # in its own process (like the original single main would have done sequentially).
    # Adjust python path or add arguments as needed.
    cmds = [
        [sys.executable, "experiments/run_phase1.py"],
        [sys.executable, "experiments/run_phase2.py"],
        [sys.executable, "experiments/run_phase3.py"]
    ]

    for cmd in cmds:
        print("Running:", " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"Command {cmd} failed with return code {ret}. Aborting full pipeline.")
            break
