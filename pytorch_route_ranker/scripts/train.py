from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import random

import torch
from torch import nn

from pytorch_route_ranker.app.config import RANKER_ROOT
from pytorch_route_ranker.app.model import AmidsRouteRanker
from pytorch_route_ranker.app.registry import load_registry, registry_fingerprint, route_text
from pytorch_route_ranker.app.text_features import HashingTextVectorizer


def parse_args():
    parser = ArgumentParser(description="Train the local AMIDS PyTorch route ranker.")
    parser.add_argument("--registry", type=Path, default=RANKER_ROOT / "data" / "route_registry.json")
    parser.add_argument(
        "--training-data",
        type=Path,
        nargs="+",
        default=[
            RANKER_ROOT / "data" / "generated_training_examples.jsonl",
            RANKER_ROOT / "data" / "expert_training_examples.jsonl",
        ],
    )
    parser.add_argument("--output", type=Path, default=RANKER_ROOT / "models" / "route_ranker.pt")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--feature-dimension", type=int, default=4096)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:N, or mps.",
    )
    return parser.parse_args()


def select_device(requested_device: str) -> torch.device:
    normalized_device = requested_device.strip().lower()
    if normalized_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized_device == "cpu":
        return torch.device("cpu")
    if normalized_device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested, but this PyTorch installation cannot use MPS.")
        return torch.device("mps")
    if normalized_device == "cuda" or normalized_device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but this PyTorch installation cannot use CUDA.")
        device = torch.device(normalized_device)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device_index} is unavailable; "
                f"detected {torch.cuda.device_count()} CUDA device(s)."
            )
        return device
    raise ValueError("Training device must be auto, cpu, cuda, cuda:N, or mps.")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{device_index} ({torch.cuda.get_device_name(device_index)})"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"


def portable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
    }


def load_examples(paths: list[Path], valid_route_ids: set[str]) -> list[dict]:
    examples: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, start=1):
                if not line.strip():
                    continue
                example = json.loads(line)
                route_ids = example.get("relevantRouteIds", [])
                invalid_ids = set(route_ids) - valid_route_ids
                if invalid_ids:
                    raise ValueError(f"{path}:{line_number} contains invalid route IDs: {invalid_ids}")
                if example.get("scope") not in {"single", "multiple"} or not route_ids:
                    raise ValueError(f"{path}:{line_number} has an invalid scope or no relevant routes.")
                examples.append(example)
    if not examples:
        raise ValueError("No training examples were found. Run npm run ranker:generate first.")
    return examples


def vectorize_examples(examples, vectorizer, route_index):
    query_vectors = vectorizer.transform([example["query"] for example in examples])
    relevance_targets = torch.zeros((len(examples), len(route_index)), dtype=torch.float32)
    scope_targets = torch.zeros(len(examples), dtype=torch.float32)

    for example_index, example in enumerate(examples):
        for route_id in example["relevantRouteIds"]:
            relevance_targets[example_index, route_index[route_id]] = 1.0
        scope_targets[example_index] = 1.0 if example["scope"] == "multiple" else 0.0
    return query_vectors, relevance_targets, scope_targets


def evaluate(model, query_vectors, relevance_targets, scope_targets, route_vectors):
    model.eval()
    with torch.inference_mode():
        relevance_logits, scope_logits = model(query_vectors, route_vectors)
        top_routes = relevance_logits.argmax(dim=1)
        top_one_accuracy = float(
            relevance_targets[
                torch.arange(len(top_routes), device=relevance_targets.device),
                top_routes,
            ]
            .mean()
            .item()
        )
        predicted_scope = (torch.sigmoid(scope_logits) >= 0.5).float()
        scope_accuracy = float((predicted_scope == scope_targets).float().mean().item())
    return top_one_accuracy, scope_accuracy


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    print(f"trainingDevice={describe_device(device)} requestedDevice={args.device}")

    routes = load_registry(args.registry)
    route_index = {route["id"]: index for index, route in enumerate(routes)}
    examples = load_examples(args.training_data, set(route_index))
    random.shuffle(examples)

    validation_count = max(1, int(len(examples) * args.validation_fraction))
    validation_examples = examples[:validation_count]
    training_examples = examples[validation_count:]

    vectorizer = HashingTextVectorizer(args.feature_dimension)
    route_vectors = vectorizer.transform([route_text(route) for route in routes]).to(device)
    train_queries, train_relevance, train_scope = vectorize_examples(
        training_examples, vectorizer, route_index
    )
    validation_queries, validation_relevance, validation_scope = vectorize_examples(
        validation_examples, vectorizer, route_index
    )
    train_queries = train_queries.to(device)
    train_relevance = train_relevance.to(device)
    train_scope = train_scope.to(device)
    validation_queries = validation_queries.to(device)
    validation_relevance = validation_relevance.to(device)
    validation_scope = validation_scope.to(device)

    model = AmidsRouteRanker(args.feature_dimension, args.hidden_dimension).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001,
    )
    relevance_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0, device=device))
    scope_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(len(training_examples), device=device)
        epoch_loss = torch.zeros((), device=device)

        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            relevance_logits, scope_logits = model(train_queries[batch_indices], route_vectors)
            relevance_loss = relevance_loss_fn(relevance_logits, train_relevance[batch_indices])
            scope_loss = scope_loss_fn(scope_logits, train_scope[batch_indices])
            loss = relevance_loss + (scope_loss * 0.5)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(batch_indices)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            top_one_accuracy, scope_accuracy = evaluate(
                model,
                validation_queries,
                validation_relevance,
                validation_scope,
                route_vectors,
            )
            average_epoch_loss = float((epoch_loss / len(training_examples)).item())
            print(
                f"epoch={epoch:03d} loss={average_epoch_loss:.4f} "
                f"validationTop1={top_one_accuracy:.3f} validationScope={scope_accuracy:.3f}"
            )

    model_version = datetime.now(timezone.utc).strftime("amids-ranker-%Y%m%dT%H%M%SZ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": portable_state_dict(model),
            "feature_dimension": args.feature_dimension,
            "hidden_dimension": args.hidden_dimension,
            "registry_fingerprint": registry_fingerprint(routes),
            "model_version": model_version,
            "training_example_count": len(examples),
            "training_device": describe_device(device),
            "torch_version": str(torch.__version__),
        },
        args.output,
    )
    print(f"Saved {model_version} to {args.output}")


if __name__ == "__main__":
    main()
