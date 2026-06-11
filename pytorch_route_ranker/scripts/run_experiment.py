from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys

from pytorch_route_ranker.app.config import PROJECT_ROOT, RANKER_ROOT
from pytorch_route_ranker.app.registry import load_registry, registry_fingerprint


DATA_ROOT = RANKER_ROOT / "data"
RUNS_ROOT = RANKER_ROOT / "runs"
SUMMARY_FIELDS = [
    "runId",
    "createdAt",
    "status",
    "notes",
    "routeCount",
    "generatedExamples",
    "expertExamples",
    "testExamples",
    "epochs",
    "batchSize",
    "learningRate",
    "featureDimension",
    "hiddenDimension",
    "trainingDevice",
    "seed",
    "scopeAccuracy",
    "topRouteAccuracy",
    "relevantRouteRecall",
    "fallbackRate",
    "averageLatencyMs",
    "multipleRouteExactSetAccuracy",
    "multipleRouteRecall",
    "eligibleForPromotion",
]
METRIC_PATTERN = re.compile(
    r"^(examples|scopeAccuracy|topRouteAccuracy|relevantRouteRecall|fallbackRate|"
    r"averageLatencyMs)=([^\s]+)$"
)


def parse_args():
    parser = ArgumentParser(
        description="Create a reproducible AMIDS ranker training and evaluation run."
    )
    parser.add_argument("--run-name", default=None, help="Optional readable suffix for the run ID.")
    parser.add_argument("--notes", default="", help="Short explanation of what changed in this run.")
    parser.add_argument(
        "--held-out-test",
        type=Path,
        default=DATA_ROOT / "held_out_test.jsonl",
        help="Held-out JSONL file used only for evaluation.",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--feature-dimension", type=int, default=4096)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device passed to train.py: auto, cpu, cuda, cuda:N, or mps.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-mismatches", type=int, default=20)
    parser.add_argument(
        "--active-model",
        type=Path,
        default=RANKER_ROOT / "models" / "route_ranker.pt",
        help="Active checkpoint used for automatic candidate comparison.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Use the existing exported route_registry.json.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Use the existing generated_training_examples.jsonl.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def make_run_id(run_name: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = safe_name(run_name) if run_name else ""
    base_id = f"{timestamp}-{suffix}" if suffix else timestamp
    candidate = base_id
    counter = 2

    while (RUNS_ROOT / candidate).exists():
        candidate = f"{base_id}-{counter}"
        counter += 1
    return candidate


def run_and_tee(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> str:
    print(f"\n> {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines: list[str] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output_lines.append(line)

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
    return "".join(output_lines)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as data_file:
        return sum(1 for line in data_file if line.strip())


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise ValueError(f"Required file does not exist: {source}")
    shutil.copy2(source, destination)


def copy_optional(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    shutil.copy2(source, destination)
    return True


def snapshot_source(source_directory: Path) -> None:
    app_source = RANKER_ROOT / "app"
    shutil.copytree(
        app_source,
        source_directory / "app",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    scripts_destination = source_directory / "scripts"
    scripts_destination.mkdir()
    for script_name in [
        "compare_models.py",
        "evaluate.py",
        "generate_training_data.py",
        "manage_model.py",
        "run_experiment.py",
        "train.py",
    ]:
        shutil.copy2(RANKER_ROOT / "scripts" / script_name, scripts_destination / script_name)


def parse_metrics(evaluation_output: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in evaluation_output.splitlines():
        match = METRIC_PATTERN.match(line.strip())
        if match:
            metrics[match.group(1)] = match.group(2)
    return metrics


def write_json(path: Path, payload: dict) -> None:
    path.write_text(f"{json.dumps(payload, indent=2, ensure_ascii=True)}\n", encoding="utf-8")


def append_summary(summary_row: dict[str, object]) -> None:
    summary_path = RUNS_ROOT / "summary.csv"
    existing_rows: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as summary_file:
            reader = csv.DictReader(summary_file)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
        if existing_fields != SUMMARY_FIELDS:
            with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
                writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(
                        {field: existing_row.get(field, "") for field in SUMMARY_FIELDS}
                    )

    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with summary_path.open("a", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: summary_row.get(field, "") for field in SUMMARY_FIELDS})


def main() -> None:
    args = parse_args()
    held_out_test = args.held_out_test.resolve()
    generated_data = DATA_ROOT / "generated_training_examples.jsonl"
    expert_data = DATA_ROOT / "expert_training_examples.jsonl"
    exported_registry = DATA_ROOT / "route_registry.json"

    if held_out_test in {generated_data.resolve(), expert_data.resolve()}:
        raise ValueError("The held-out test file must never also be used as training data.")

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args.run_name)
    run_directory = RUNS_ROOT / run_id
    snapshot_directory = run_directory / "data"
    source_directory = run_directory / "source"
    snapshot_directory.mkdir(parents=True)
    snapshot_source(source_directory)

    status = "failed"
    metrics: dict[str, str] = {}
    configuration: dict[str, object] = {
        "runId": run_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "notes": args.notes,
        "pythonExecutable": sys.executable,
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "nodeVersion": command_output(["node", "--version"]),
        "gitCommit": command_output(["git", "rev-parse", "HEAD"]),
        "gitStatus": command_output(["git", "status", "--short"]),
        "training": {
            "epochs": args.epochs,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "featureDimension": args.feature_dimension,
            "hiddenDimension": args.hidden_dimension,
            "requestedDevice": args.device,
            "validationFraction": args.validation_fraction,
            "seed": args.seed,
        },
    }
    write_json(run_directory / "configuration.json", configuration)
    (run_directory / "notes.txt").write_text(f"{args.notes}\n", encoding="utf-8")

    try:
        if not args.skip_export:
            run_and_tee(
                ["node", "pytorch_route_ranker/scripts/export_route_registry.mjs"],
                run_directory / "export-log.txt",
            )
        if not args.skip_generate:
            run_and_tee(
                [sys.executable, "-m", "pytorch_route_ranker.scripts.generate_training_data"],
                run_directory / "generation-log.txt",
            )

        snapshot_registry = snapshot_directory / "route_registry.json"
        snapshot_generated = snapshot_directory / "generated_training_examples.jsonl"
        snapshot_expert = snapshot_directory / "expert_training_examples.jsonl"
        snapshot_test = snapshot_directory / "held_out_test.jsonl"
        copy_required(exported_registry, snapshot_registry)
        copy_required(generated_data, snapshot_generated)
        has_expert_data = copy_optional(expert_data, snapshot_expert)
        copy_required(held_out_test, snapshot_test)

        training_data = [snapshot_generated]
        if has_expert_data and line_count(snapshot_expert) > 0:
            training_data.append(snapshot_expert)

        registry = load_registry(snapshot_registry)
        data_manifest = {
            path.name: {
                "lineCount": line_count(path),
                "sha256": file_sha256(path),
            }
            for path in [snapshot_registry, snapshot_generated, snapshot_test, *training_data[1:]]
        }
        configuration["registryFingerprint"] = registry_fingerprint(registry)
        configuration["routeCount"] = len(registry)
        configuration["dataManifest"] = data_manifest
        write_json(run_directory / "configuration.json", configuration)

        model_path = run_directory / "model.pt"
        training_command = [
            sys.executable,
            "-m",
            "pytorch_route_ranker.scripts.train",
            "--registry",
            str(snapshot_registry),
            "--training-data",
            *[str(path) for path in training_data],
            "--output",
            str(model_path),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--feature-dimension",
            str(args.feature_dimension),
            "--hidden-dimension",
            str(args.hidden_dimension),
            "--device",
            args.device,
            "--validation-fraction",
            str(args.validation_fraction),
            "--seed",
            str(args.seed),
        ]
        training_output = run_and_tee(training_command, run_directory / "training-log.txt")
        training_device_match = re.search(r"^trainingDevice=(.+?) requestedDevice=", training_output, re.MULTILINE)
        configuration["training"]["selectedDevice"] = (
            training_device_match.group(1) if training_device_match else "unknown"
        )

        evaluation_env = os.environ.copy()
        evaluation_env["AMIDS_RANKER_CHECKPOINT_PATH"] = str(model_path)
        evaluation_env["AMIDS_RANKER_REGISTRY_PATH"] = str(snapshot_registry)
        evaluation_env["AMIDS_RANKER_FEATURE_DIMENSION"] = str(args.feature_dimension)
        evaluation_results_path = run_directory / "evaluation-results.json"
        evaluation_output = run_and_tee(
            [
                sys.executable,
                "-m",
                "pytorch_route_ranker.scripts.evaluate",
                "--data",
                str(snapshot_test),
                "--show-mismatches",
                str(args.show_mismatches),
                "--output",
                str(evaluation_results_path),
            ],
            run_directory / "evaluation-log.txt",
            env=evaluation_env,
        )
        metrics = parse_metrics(evaluation_output)
        evaluation_results = json.loads(evaluation_results_path.read_text(encoding="utf-8"))
        metrics["multipleRouteExactSetAccuracy"] = str(
            evaluation_results["multipleRoute"]["exactSetAccuracy"]
        )
        metrics["multipleRouteRecall"] = str(
            evaluation_results["multipleRoute"]["relevantRouteRecall"]
        )

        active_model = args.active_model.resolve()
        comparison_path = run_directory / "comparison.json"
        if active_model.exists():
            try:
                run_and_tee(
                    [
                        sys.executable,
                        "-m",
                        "pytorch_route_ranker.scripts.compare_models",
                        "--current",
                        str(active_model),
                        "--candidate",
                        str(model_path),
                        "--data",
                        str(snapshot_test),
                        "--registry",
                        str(snapshot_registry),
                        "--output",
                        str(comparison_path),
                    ],
                    run_directory / "comparison-log.txt",
                )
                comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                configuration["comparison"] = {
                    "status": "complete",
                    "eligibleForPromotion": comparison["eligibleForPromotion"],
                    "failedChecks": comparison["failedChecks"],
                }
                metrics["eligibleForPromotion"] = str(comparison["eligibleForPromotion"]).lower()
            except Exception as comparison_error:
                configuration["comparison"] = {
                    "status": "failed",
                    "error": str(comparison_error),
                    "eligibleForPromotion": False,
                }
                metrics["eligibleForPromotion"] = "false"
        else:
            configuration["comparison"] = {
                "status": "not-run",
                "reason": "No active model exists yet.",
                "eligibleForPromotion": None,
            }
            metrics["eligibleForPromotion"] = ""
        status = "complete"
    except Exception as error:
        configuration["error"] = str(error)
        raise
    finally:
        configuration["status"] = status
        configuration["completedAt"] = datetime.now(timezone.utc).isoformat()
        configuration["metrics"] = metrics
        write_json(run_directory / "configuration.json", configuration)
        append_summary(
            {
                "runId": run_id,
                "createdAt": configuration["createdAt"],
                "status": status,
                "notes": args.notes,
                "routeCount": configuration.get("routeCount", ""),
                "generatedExamples": line_count(snapshot_directory / "generated_training_examples.jsonl"),
                "expertExamples": line_count(snapshot_directory / "expert_training_examples.jsonl"),
                "testExamples": line_count(snapshot_directory / "held_out_test.jsonl"),
                "epochs": args.epochs,
                "batchSize": args.batch_size,
                "learningRate": args.learning_rate,
                "featureDimension": args.feature_dimension,
                "hiddenDimension": args.hidden_dimension,
                "trainingDevice": configuration.get("training", {}).get("selectedDevice", ""),
                "seed": args.seed,
                **metrics,
            }
        )
        print(f"\nRun status: {status}")
        print(f"Run directory: {run_directory}")
        print(f"Comparison summary: {RUNS_ROOT / 'summary.csv'}")


if __name__ == "__main__":
    main()
