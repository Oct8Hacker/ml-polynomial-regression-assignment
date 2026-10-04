"""Regenerate submission predictions from a trusted saved pipeline, without fitting."""
import argparse
from pathlib import Path

import joblib

from polynomial_regression import read_features, write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trusted model.joblib from train.py")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    test = read_features(args.test, artifact["features"])
    prediction = artifact["pipeline"].predict(test.to_numpy())
    write_predictions(args.output, prediction)
    print(f"Wrote {len(prediction)} predictions to {args.output}")


if __name__ == "__main__":
    main()
