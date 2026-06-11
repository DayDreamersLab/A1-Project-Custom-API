from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import shutil

import torch

from pytorch_route_ranker.app.config import RANKER_ROOT, Settings
from pytorch_route_ranker.app.registry import load_registry, registry_fingerprint


DEFAULT_MODELS_DIRECTORY = RANKER_ROOT / "models"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-") or "model"


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary_path.write_bytes(contents)
    os.replace(temporary_path, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_bytes(
        path,
        f"{json.dumps(payload, indent=2, ensure_ascii=True)}\n".encode("utf-8"),
    )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary_path)
    os.replace(temporary_path, destination)


def append_history(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as history_file:
        history_file.write(f"{json.dumps(payload, ensure_ascii=True)}\n")
        history_file.flush()
        os.fsync(history_file.fileno())


def checkpoint_metadata(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "sha256": file_sha256(path),
        "modelVersion": str(checkpoint.get("model_version", "unknown")),
        "registryFingerprint": str(checkpoint.get("registry_fingerprint", "")),
        "featureDimension": int(checkpoint["feature_dimension"]),
        "hiddenDimension": int(checkpoint["hidden_dimension"]),
        "trainingExampleCount": int(checkpoint.get("training_example_count", 0)),
        "trainingDevice": str(checkpoint.get("training_device", "unknown")),
        "torchVersion": str(checkpoint.get("torch_version", "unknown")),
    }


def unique_release_directory(releases_directory: Path, base_id: str) -> tuple[str, Path]:
    release_id = base_id
    suffix = 2
    while (releases_directory / release_id).exists():
        release_id = f"{base_id}-{suffix}"
        suffix += 1
    return release_id, releases_directory / release_id


def archive_checkpoint(
    checkpoint_path: Path,
    releases_directory: Path,
    release_name: str,
    provenance: dict,
) -> str:
    release_id, release_directory = unique_release_directory(
        releases_directory,
        f"{timestamp_id()}-{safe_name(release_name)}",
    )
    release_directory.mkdir(parents=True)
    archived_model = release_directory / "model.pt"
    shutil.copy2(checkpoint_path, archived_model)
    metadata = checkpoint_metadata(archived_model)
    manifest = {
        "schemaVersion": 1,
        "releaseId": release_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": metadata,
        "provenance": provenance,
    }
    atomic_write_json(release_directory / "manifest.json", manifest)
    return release_id


def load_active_release_id(active_release_path: Path) -> str | None:
    if not active_release_path.exists():
        return None
    return read_json(active_release_path).get("releaseId")


def validate_registry(checkpoint: dict, registry_path: Path) -> None:
    active_fingerprint = registry_fingerprint(load_registry(registry_path))
    if checkpoint["registryFingerprint"] != active_fingerprint:
        raise ValueError(
            "Checkpoint registry fingerprint does not match the currently exported route registry."
        )


def common_paths(models_directory: Path) -> dict[str, Path]:
    return {
        "active": models_directory / "route_ranker.pt",
        "activeRelease": models_directory / "active_release.json",
        "history": models_directory / "promotion_history.jsonl",
        "releases": models_directory / "releases",
    }


def validate_approval(approved_by: str, reason: str) -> None:
    if not approved_by.strip():
        raise ValueError("Approval requires a non-empty reviewer identity.")
    if not reason.strip():
        raise ValueError("Approval requires a non-empty reason.")


def promote(args) -> None:
    validate_approval(args.approved_by, args.reason)
    paths = common_paths(args.models_directory.resolve())
    candidate_path = args.candidate.resolve()
    registry_path = args.registry.resolve()
    candidate_metadata = checkpoint_metadata(candidate_path)
    validate_registry(candidate_metadata, registry_path)

    previous_release_id = load_active_release_id(paths["activeRelease"])
    comparison_path = args.comparison.resolve() if args.comparison else None
    comparison = read_json(comparison_path) if comparison_path else None
    if paths["active"].exists():
        if not comparison:
            raise ValueError("A candidate-versus-current comparison is required for promotion.")
        if comparison.get("eligibleForPromotion") is not True:
            raise ValueError("Comparison report does not approve this candidate for promotion.")
        if (
            comparison.get("failedChecks")
            or not comparison.get("checks")
            or not all(check.get("passed") is True for check in comparison["checks"])
        ):
            raise ValueError("Comparison report contains failed or incomplete promotion checks.")
        expected_candidate_hash = comparison.get("candidate", {}).get("model", {}).get(
            "checkpointSha256"
        )
        if candidate_metadata["sha256"] != expected_candidate_hash:
            raise ValueError("Candidate checkpoint has changed since the comparison was created.")
        current_metadata = checkpoint_metadata(paths["active"])
        expected_current_hash = comparison.get("current", {}).get("model", {}).get(
            "checkpointSha256"
        )
        if current_metadata["sha256"] != expected_current_hash:
            raise ValueError(
                "Active checkpoint has changed since comparison. Run candidate comparison again."
            )
        if not previous_release_id:
            previous_release_id = archive_checkpoint(
                paths["active"],
                paths["releases"],
                "baseline",
                {"action": "baseline-before-first-managed-promotion"},
            )
    else:
        if not args.allow_initial or not args.initial_evaluation:
            raise ValueError(
                "No active model exists. Bootstrap the first release with "
                "--allow-initial and --initial-evaluation."
            )
        initial_evaluation_path = args.initial_evaluation.resolve()
        initial_evaluation = read_json(initial_evaluation_path)
        expected_candidate_hash = initial_evaluation.get("model", {}).get("checkpointSha256")
        if candidate_metadata["sha256"] != expected_candidate_hash:
            raise ValueError(
                "Candidate checkpoint has changed since the initial evaluation was created."
            )

    release_id = archive_checkpoint(
        candidate_path,
        paths["releases"],
        args.release_name or candidate_metadata["modelVersion"],
        {
            "action": "promotion-candidate",
            "comparisonPath": str(comparison_path) if comparison_path else None,
            "comparisonSha256": file_sha256(comparison_path) if comparison_path else None,
            "initialEvaluationPath": (
                str(args.initial_evaluation.resolve()) if args.initial_evaluation else None
            ),
            "initialEvaluationSha256": (
                file_sha256(args.initial_evaluation.resolve()) if args.initial_evaluation else None
            ),
            "approvedBy": args.approved_by,
            "reason": args.reason,
        },
    )
    release_directory = paths["releases"] / release_id
    if comparison_path:
        shutil.copy2(comparison_path, release_directory / "comparison.json")
    if args.initial_evaluation:
        shutil.copy2(args.initial_evaluation.resolve(), release_directory / "evaluation-results.json")
    archived_candidate = release_directory / "model.pt"
    atomic_copy(archived_candidate, paths["active"])
    atomic_write_json(
        paths["activeRelease"],
        {
            "releaseId": release_id,
            "activatedAt": datetime.now(timezone.utc).isoformat(),
        },
    )
    append_history(
        paths["history"],
        {
            "action": "promote",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "approvedBy": args.approved_by,
            "reason": args.reason,
            "fromReleaseId": previous_release_id,
            "toReleaseId": release_id,
            "modelSha256": candidate_metadata["sha256"],
            "comparisonSha256": file_sha256(comparison_path) if comparison_path else None,
            "initialEvaluationSha256": (
                file_sha256(args.initial_evaluation.resolve()) if args.initial_evaluation else None
            ),
        },
    )
    print(f"Promoted release {release_id}")
    print("Restart the PyTorch ranker API to load the promoted checkpoint.")


def rollback(args) -> None:
    validate_approval(args.approved_by, args.reason)
    paths = common_paths(args.models_directory.resolve())
    registry_path = args.registry.resolve()
    releases_directory = paths["releases"].resolve()
    release_directory = (releases_directory / args.release).resolve()
    if release_directory.parent != releases_directory:
        raise ValueError("Release ID must identify one archived release directory.")
    release_model = release_directory / "model.pt"
    manifest_path = release_directory / "manifest.json"
    if not release_model.exists() or not manifest_path.exists():
        raise ValueError(f"Release does not exist or is incomplete: {args.release}")

    manifest = read_json(manifest_path)
    release_metadata = checkpoint_metadata(release_model)
    if release_metadata["sha256"] != manifest.get("model", {}).get("sha256"):
        raise ValueError("Archived release checkpoint does not match its immutable manifest.")
    validate_registry(release_metadata, registry_path)

    previous_release_id = load_active_release_id(paths["activeRelease"])
    atomic_copy(release_model, paths["active"])
    atomic_write_json(
        paths["activeRelease"],
        {
            "releaseId": args.release,
            "activatedAt": datetime.now(timezone.utc).isoformat(),
        },
    )
    append_history(
        paths["history"],
        {
            "action": "rollback",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "approvedBy": args.approved_by,
            "reason": args.reason,
            "fromReleaseId": previous_release_id,
            "toReleaseId": args.release,
            "modelSha256": release_metadata["sha256"],
        },
    )
    print(f"Rolled back to release {args.release}")
    print("Restart the PyTorch ranker API to load the rollback checkpoint.")


def status(args) -> None:
    paths = common_paths(args.models_directory.resolve())
    active_release_id = load_active_release_id(paths["activeRelease"])
    print(f"activeRelease={active_release_id or 'unmanaged-or-none'}")
    if paths["active"].exists():
        metadata = checkpoint_metadata(paths["active"])
        print(f"activeModelVersion={metadata['modelVersion']}")
        print(f"activeModelSha256={metadata['sha256']}")
    else:
        print("activeModel=missing")
    releases = sorted(
        path.name for path in paths["releases"].iterdir() if path.is_dir()
    ) if paths["releases"].exists() else []
    print(f"availableReleases={','.join(releases)}")


def parse_args():
    parser = ArgumentParser(description="Promote and roll back approved AMIDS ranker releases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument(
            "--models-directory",
            type=Path,
            default=DEFAULT_MODELS_DIRECTORY,
        )
        subparser.add_argument(
            "--registry",
            type=Path,
            default=Settings().registry_path,
        )

    promote_parser = subparsers.add_parser("promote")
    add_common(promote_parser)
    promote_parser.add_argument("--candidate", type=Path, required=True)
    promote_parser.add_argument("--comparison", type=Path, default=None)
    promote_parser.add_argument("--allow-initial", action="store_true")
    promote_parser.add_argument("--initial-evaluation", type=Path, default=None)
    promote_parser.add_argument("--approved-by", required=True)
    promote_parser.add_argument("--reason", required=True)
    promote_parser.add_argument("--release-name", default=None)

    rollback_parser = subparsers.add_parser("rollback")
    add_common(rollback_parser)
    rollback_parser.add_argument("--release", required=True)
    rollback_parser.add_argument("--approved-by", required=True)
    rollback_parser.add_argument("--reason", required=True)

    status_parser = subparsers.add_parser("status")
    add_common(status_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "promote":
        promote(args)
    elif args.command == "rollback":
        rollback(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
