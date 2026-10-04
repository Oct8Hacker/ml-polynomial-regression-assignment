"""Train both assignment models; rerunning an identical command resumes checkpoints."""
import argparse
from pathlib import Path

from polynomial_regression import read_json, train_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Traning-testing-data"))
    parser.add_argument("--rollno", default="BT2024151")
    parser.add_argument("--problems", type=int, choices=[1, 2], nargs="+", default=[1, 2])
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--jobs", type=int, default=2, help="Parallel degrees; keep low to limit RAM use")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    config = read_json(args.config)
    print(f"Config: {args.config}; output: {args.output}; jobs: {args.jobs}", flush=True)
    for problem in args.problems:
        print(f"\nProblem {problem}", flush=True)
        train_problem(args.data_dir, args.rollno, problem, config, args.output, args.jobs)


if __name__ == "__main__":
    main()
