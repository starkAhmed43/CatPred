import argparse
import json
from pathlib import Path

import optuna


def main():
    parser = argparse.ArgumentParser(description="Export best Optuna trial params to JSON.")
    parser.add_argument("--storage", required=True, type=str)
    parser.add_argument("--study_name", required=True, type=str)
    parser.add_argument("--out_json", required=True, type=str)
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "study_name": args.study_name,
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "direction": study.direction.name.lower(),
        "params": dict(study.best_trial.params),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved best trial params to: {out_path}")


if __name__ == "__main__":
    main()
