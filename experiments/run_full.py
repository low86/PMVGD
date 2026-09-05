"""Run the complete PMVGD pipeline: Phase 1 -> Phase 2 -> Phase 3."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all three PMVGD training phases with shared settings."
    )
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], default="mimic3")
    parser.add_argument("--mimic3_path", type=str, default="")
    parser.add_argument("--mimic4_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="./cache")
    parser.add_argument("--model", type=str, default="teacher")
    parser.add_argument("--dev", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch_main", type=int, default=130)
    parser.add_argument("--epoch_view", type=int, default=20)
    parser.add_argument("--epoch_kd", type=int, default=50)
    parser.add_argument("--epoch_test", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.mimic3_path if args.dataset == "mimic3" else args.mimic4_path
    if not dataset_path:
        raise SystemExit(f"--{args.dataset}_path is required when --dataset {args.dataset}.")

    common_args = [
        "--dataset", args.dataset,
        "--mimic3_path", args.mimic3_path,
        "--mimic4_path", args.mimic4_path,
        "--data_path", args.data_path,
        "--model", args.model,
        "--dev", str(args.dev),
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--hidden_size", str(args.hidden_size),
        "--lr", str(args.lr),
        "--epoch_test", str(args.epoch_test),
    ]

    ckpt_root = PROJECT_ROOT / "ckpt"
    phase1_ckpt = ckpt_root / "phase1" / f"phase1_{args.model}_{args.dataset}_{args.seed}.ckpt"
    phase2_ckpt = ckpt_root / "phase2" / f"phase2_pred_{args.model}_{args.dataset}_{args.seed}.ckpt"
    phase3_ckpt = ckpt_root / "phase3" / f"phase3_{args.model}_{args.dataset}_{args.seed}.ckpt"
    phases = [
        ("Phase 1", "experiments.run_phase1", ["--epoch_main", str(args.epoch_main)], phase1_ckpt),
        ("Phase 2", "experiments.run_phase2", ["--epoch_view", str(args.epoch_view)], phase2_ckpt),
        ("Phase 3", "experiments.run_phase3", ["--epoch_kd", str(args.epoch_kd)], phase3_ckpt),
    ]

    for phase_name, module, phase_args, expected_ckpt in phases:
        cmd = [sys.executable, "-m", module, *common_args, *phase_args]
        print(f"Running {phase_name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        if not expected_ckpt.exists():
            raise RuntimeError(
                f"{phase_name} exited without creating its expected checkpoint: {expected_ckpt}"
            )


if __name__ == "__main__":
    main()
