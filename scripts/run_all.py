#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="ROBIN Benchmark - Run All Phases")
    parser.add_argument("--config", type=str, default="configs/full_config.yaml")
    parser.add_argument("--skip-inference", action="store_true", help="Skip Phase 2 (inference)")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge in Phase 3")
    parser.add_argument("--sample-size", type=int, default=None, help="Override sample size")
    args = parser.parse_args()
    
    scripts_dir = Path(__file__).parent
    
    print("=" * 60)
    print("ROBIN Benchmark - Full Pipeline")
    print("=" * 60)
    
    print("\n[Phase 1] Dataset Construction...")
    cmd = [sys.executable, str(scripts_dir / "run_phase1.py"), "--config", args.config]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Phase 1 failed!")
        sys.exit(1)
    
    if not args.skip_inference:
        print("\n[Phase 2] Model Inference...")
        cmd = [sys.executable, str(scripts_dir / "run_phase2.py"), "--config", args.config]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("Phase 2 failed!")
            sys.exit(1)
    else:
        print("\n[Phase 2] Skipped (--skip-inference)")
    
    print("\n[Phase 3] Evaluation...")
    cmd = [sys.executable, str(scripts_dir / "run_phase3.py"), "--config", args.config]
    if not args.skip_judge:
        cmd.append("--run-judge")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Phase 3 failed!")
        sys.exit(1)
    
    print("\n[Phase 4] Analysis & Reporting...")
    cmd = [sys.executable, str(scripts_dir / "run_phase4.py"), "--config", args.config]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Phase 4 failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("ROBIN Benchmark Complete!")
    print("Results saved to: results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
