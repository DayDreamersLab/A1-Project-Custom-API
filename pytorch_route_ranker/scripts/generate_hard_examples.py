from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import os
import re
import time

from pytorch_route_ranker.app.config import RANKER_ROOT
from pytorch_route_ranker.app.registry import load_registry, registry_fingerprint


DATA_ROOT = RANKER_ROOT / "data"
DEFAULT_EVIDENCE_PATH = DATA_ROOT / "reviewable_interaction_evidence.jsonl"
DEFAULT_REGISTRY_PATH = DATA_ROOT / "route_registry.json"
DEFAULT_OUTPUT_PATH = DATA_ROOT / "hard_example_training_data.jsonl"
DEFAULT_MANIFEST_PATH = DATA_ROOT / "hard_example_training_manifest.json"
DEFAULT_REFERENCE_PATHS = [
    DATA_ROOT / "generated_training_examples.jsonl",
    DATA_ROOT / "expert_training_examples.jsonl",
]
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
MULTIPLE_PATTERN = re.compile(
    r"\b(?:all|every|everything|each|complete|comprehensive)\b|"
    r"\bfull\s+(?:set|range|collection)\b|\bany\s+and\s+all\b",
    re.IGNORECASE,
)
QUALIFIER_GROUPS = {
    "current": {"current", "latest", "now", "real time", "realtime", "live"},
    "forecast": {"forecast", "predicted", "expected", "outlook", "future"},
    "historical": {"historical", "history", "past", "archive", "archived"},
    "alert": {"alert", "alerts", "warning", "warnings", "advisory", "advisories"},
}
NEGATION_TERMS = {"not", "without", "exclude", "excluding", "except", "only", "just"}


def parse_args():
    parser = ArgumentParser(
        description=(
            "Generate controlled hard-example paraphrases from approved interaction corrections."
        )
    )
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--reference-data",
        type=Path,
        nargs="*",
        default=DEFAULT_REFERENCE_PATHS,
        help="Existing training JSONL files checked for duplicate or conflicting queries.",
    )
    parser.add_argument(
        "--model",
        default=(
            os.getenv("HARD_EXAMPLE_LLM_MODEL")
            or os.getenv("OLLAMA_MODEL")
            or "qwen3:latest"
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default=(
            os.getenv("HARD_EXAMPLE_OLLAMA_URL")
            or os.getenv("OLLAMA_URL")
            or "http://127.0.0.1:11434/api/chat"
        ),
    )
    parser.add_argument(
        "--validator-model",
        default=os.getenv("HARD_EXAMPLE_VALIDATOR_MODEL"),
        help="Optional stronger local model used for semantic validation.",
    )
    parser.add_argument(
        "--generate-count",
        type=int,
        default=30,
        help="Number of candidate paraphrases requested before validation.",
    )
    parser.add_argument(
        "--max-paraphrases",
        type=int,
        default=15,
        help="Maximum accepted paraphrases retained per approved correction.",
    )
    parser.add_argument(
        "--limit-corrections",
        type=int,
        default=0,
        help="Maximum approved corrections to process; zero processes all.",
    )
    parser.add_argument(
        "--evidence-id",
        action="append",
        default=[],
        help="Process only this evidence ID; may be supplied more than once.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-semantic-validation",
        action="store_true",
        help="Accept deterministic checks without the second local-LLM semantic validation pass.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate approvals and show the generation plan without calling Ollama or writing files.",
    )
    return parser.parse_args()


def normalize_query(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(value).lower()))


def unique_strings(values) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if value is not None and str(value).strip()
        )
    )


def label_key(scope: str, route_ids: list[str]) -> tuple[str, tuple[str, ...]]:
    return scope, tuple(sorted(set(route_ids)))


def read_jsonl(path: Path, required: bool = False) -> list[dict]:
    if not path.exists():
        if required:
            raise ValueError(f"Required JSONL file does not exist: {path}")
        return []

    records = []
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} contains invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain one JSON object.")
            records.append(record)
    return records


def validate_training_record(record: dict, valid_route_ids: set[str], source: str) -> None:
    query = record.get("query")
    scope = record.get("scope")
    route_ids = record.get("relevantRouteIds")
    if not isinstance(query, str) or not normalize_query(query):
        raise ValueError(f"{source} contains a record without a valid query.")
    if scope not in {"single", "multiple"}:
        raise ValueError(f"{source} query {query!r} has an invalid scope.")
    if not isinstance(route_ids, list) or not route_ids:
        raise ValueError(f"{source} query {query!r} has no relevant route IDs.")
    invalid_ids = set(route_ids) - valid_route_ids
    if invalid_ids:
        raise ValueError(f"{source} query {query!r} contains invalid route IDs: {invalid_ids}")
    if scope == "single" and len(set(route_ids)) != 1:
        raise ValueError(f"{source} query {query!r} is single but has multiple route IDs.")
    if scope == "multiple" and len(set(route_ids)) < 2:
        raise ValueError(f"{source} query {query!r} is multiple but has fewer than two routes.")


def load_approved_corrections(
    evidence_path: Path,
    valid_route_ids: set[str],
    selected_evidence_ids: set[str],
) -> tuple[list[dict], set[str]]:
    evidence_records = read_jsonl(evidence_path, required=True)
    known_evidence_ids = {
        str(record.get("evidenceId"))
        for record in evidence_records
        if record.get("evidenceId") is not None
    }
    corrections = []

    for record in evidence_records:
        evidence_id = str(record.get("evidenceId", "")).strip()
        if selected_evidence_ids and evidence_id not in selected_evidence_ids:
            continue
        if record.get("reviewStatus") != "approved":
            continue

        correction = {
            "evidenceId": evidence_id,
            "query": str(record.get("query", "")).strip(),
            "scope": record.get("approvedScope"),
            "relevantRouteIds": unique_strings(record.get("approvedRelevantRouteIds", [])),
            "suggestedRouteIds": unique_strings(record.get("suggestedRouteIds", [])),
            "reviewedAt": record.get("reviewedAt"),
        }
        if not evidence_id:
            raise ValueError("An approved interaction correction has no evidenceId.")
        validate_training_record(correction, valid_route_ids, f"approved evidence {evidence_id}")
        corrections.append(correction)

    if selected_evidence_ids:
        missing_ids = selected_evidence_ids - known_evidence_ids
        if missing_ids:
            raise ValueError(f"Requested evidence IDs were not found: {sorted(missing_ids)}")
    return corrections, known_evidence_ids


def deduplicate_corrections(corrections: list[dict]) -> tuple[list[dict], int]:
    unique_corrections = []
    labels_by_query: dict[str, tuple[str, tuple[str, ...]]] = {}
    duplicate_count = 0
    for correction in corrections:
        normalized = normalize_query(correction["query"])
        label = label_key(correction["scope"], correction["relevantRouteIds"])
        previous_label = labels_by_query.get(normalized)
        if previous_label and previous_label != label:
            raise ValueError(
                f"Approved corrections assign conflicting labels to query {correction['query']!r}."
            )
        if previous_label:
            duplicate_count += 1
            continue
        labels_by_query[normalized] = label
        unique_corrections.append(correction)
    return unique_corrections, duplicate_count


def compact_route(route: dict) -> dict:
    return {
        "id": route["id"],
        "title": route["title"],
        "description": route["description"],
        "keywords": route["keywords"],
    }


def contains_term(normalized: str, terms: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms)


def deterministic_rejection_reason(
    candidate: str,
    correction: dict,
    valid_route_ids: set[str],
) -> str | None:
    normalized = normalize_query(candidate)
    original_normalized = normalize_query(correction["query"])
    words = normalized.split()
    if not normalized or normalized == original_normalized:
        return "empty-or-original-query"
    if len(candidate) > 240 or len(words) > 40:
        return "query-too-long"
    if URL_PATTERN.search(candidate):
        return "contains-url"
    internal_style_route_ids = [
        route_id for route_id in valid_route_ids if re.search(r"[-_/]", route_id)
    ]
    if any(route_id.lower() in candidate.lower() for route_id in internal_style_route_ids):
        return "contains-internal-route-id"

    requests_multiple = bool(MULTIPLE_PATTERN.search(normalized))
    if correction["scope"] == "single" and requests_multiple:
        return "changed-single-query-to-multiple"
    if correction["scope"] == "multiple" and not requests_multiple:
        return "lost-explicit-multiple-scope"

    for qualifier_name, qualifier_terms in QUALIFIER_GROUPS.items():
        if contains_term(original_normalized, qualifier_terms) and not contains_term(
            normalized, qualifier_terms
        ):
            return f"lost-{qualifier_name}-qualifier"
    if contains_term(original_normalized, NEGATION_TERMS) and not contains_term(
        normalized, NEGATION_TERMS
    ):
        return "lost-negation-or-exclusion"
    return None


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(normalize_query(left).split())
    right_tokens = set(normalize_query(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def remove_near_duplicates(candidates: list[str], existing_queries: list[str]) -> tuple[list[str], int]:
    accepted = []
    rejected = 0
    comparisons = list(existing_queries)
    for candidate in candidates:
        if any(jaccard_similarity(candidate, existing) >= 0.9 for existing in comparisons):
            rejected += 1
            continue
        accepted.append(candidate)
        comparisons.append(candidate)
    return accepted, rejected


def extract_json(content: str) -> dict:
    content = str(content).strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Ollama did not return a JSON object.")
        result = json.loads(content[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Ollama returned JSON that was not an object.")
    return result


def call_ollama(
    *,
    url: str,
    model: str,
    messages: list[dict],
    schema: dict,
    temperature: float,
    num_predict: int,
    timeout_seconds: float,
    retries: int,
    seed: int,
) -> tuple[dict, dict]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "format": schema,
        "options": {
            "temperature": temperature,
            "num_ctx": 4096,
            "num_predict": num_predict,
            "seed": seed,
        },
        "messages": messages,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error = None

    for attempt in range(retries + 1):
        started_at = time.perf_counter()
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            content = response_data.get("message", {}).get("content", "")
            result = extract_json(content)
            return result, {
                "roundTripMs": round((time.perf_counter() - started_at) * 1000, 2),
                "promptEvalCount": response_data.get("prompt_eval_count"),
                "evalCount": response_data.get("eval_count"),
            }
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:500]
            last_error = RuntimeError(f"Ollama returned HTTP {error.code}: {body}")
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as error:
            last_error = error
        if attempt < retries:
            time.sleep(min(2**attempt, 4))

    raise RuntimeError(f"Local Ollama request failed after {retries + 1} attempts: {last_error}")


def generate_candidates(args, correction: dict, routing_context: dict, seed: int) -> tuple[list[str], dict]:
    schema = {
        "type": "object",
        "properties": {
            "paraphrases": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": args.generate_count,
            }
        },
        "required": ["paraphrases"],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You write varied but semantically equivalent aviation navigation queries. "
                "Preserve the exact request scope, subject, location, time meaning, exclusions, "
                "and operational intent. Do not add facts, broaden the request, mention internal "
                "route IDs, include rejected suggested routes, or answer the query. Use realistic "
                "commands, questions, fragments, abbreviations, and natural user wording. Return "
                "only the required JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": f"Generate up to {args.generate_count} distinct equivalent queries.",
                    "originalQuery": correction["query"],
                    "requiredScope": correction["scope"],
                    **routing_context,
                },
                ensure_ascii=True,
            ),
        },
    ]
    result, metrics = call_ollama(
        url=args.ollama_url,
        model=args.model,
        messages=messages,
        schema=schema,
        temperature=0.55,
        num_predict=max(600, args.generate_count * 35),
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        seed=seed,
    )
    paraphrases = result.get("paraphrases")
    if not isinstance(paraphrases, list):
        raise ValueError("Ollama generation response did not contain a paraphrases array.")
    return unique_strings(paraphrases), metrics


def semantic_validation(
    args,
    correction: dict,
    routing_context: dict,
    candidates: list[str],
    seed: int,
) -> tuple[list[str], dict]:
    if not candidates:
        return [], {"skipped": True, "reason": "no-candidates-passed-deterministic-checks"}
    if args.skip_semantic_validation:
        return candidates, {"skipped": True}

    schema = {
        "type": "object",
        "properties": {
            "acceptedIndexes": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": max(0, len(candidates) - 1)},
                "uniqueItems": True,
            }
        },
        "required": ["acceptedIndexes"],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict semantic-equivalence reviewer for aviation navigation queries. "
                "Accept a candidate only if it asks for exactly the same information and approved "
                "route set as the original. Reject changes to single-versus-multiple scope, time "
                "meaning, location, flight phase, inclusion, exclusion, or operational intent. "
                "Reject candidates that also imply any rejected suggested route. When uncertain, "
                "reject it. Return only indexes from the supplied candidate list."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "originalQuery": correction["query"],
                    "requiredScope": correction["scope"],
                    **routing_context,
                    "candidates": [
                        {"index": index, "query": candidate}
                        for index, candidate in enumerate(candidates)
                    ],
                },
                ensure_ascii=True,
            ),
        },
    ]
    result, metrics = call_ollama(
        url=args.ollama_url,
        model=args.validator_model,
        messages=messages,
        schema=schema,
        temperature=0,
        num_predict=max(160, len(candidates) * 5),
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        seed=seed,
    )
    indexes = result.get("acceptedIndexes")
    if not isinstance(indexes, list):
        raise ValueError("Ollama validation response did not contain acceptedIndexes.")
    accepted_indexes = {
        index
        for index in indexes
        if isinstance(index, int) and 0 <= index < len(candidates)
    }
    return [
        candidate for index, candidate in enumerate(candidates) if index in accepted_indexes
    ], metrics


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    contents = "".join(f"{json.dumps(record, ensure_ascii=True)}\n" for record in records)
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(
        f"{json.dumps(payload, indent=2, ensure_ascii=True)}\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    args.validator_model = args.validator_model or args.model
    if args.generate_count < 1:
        raise ValueError("--generate-count must be at least 1.")
    if args.max_paraphrases < 1:
        raise ValueError("--max-paraphrases must be at least 1.")

    routes = load_registry(args.registry)
    routes_by_id = {route["id"]: route for route in routes}
    valid_route_ids = set(routes_by_id)
    fingerprint = registry_fingerprint(routes)
    selected_evidence_ids = set(args.evidence_id)
    all_approved_corrections, known_evidence_ids = load_approved_corrections(
        args.evidence,
        valid_route_ids,
        selected_evidence_ids,
    )
    approved_corrections, duplicate_correction_count = deduplicate_corrections(
        all_approved_corrections
    )
    approved_ids_for_sync = {
        correction["evidenceId"] for correction in approved_corrections
    }
    corrections = approved_corrections
    if args.limit_corrections > 0:
        corrections = corrections[: args.limit_corrections]

    existing_records = read_jsonl(args.output)
    retained_records = []
    selected_or_known_ids = selected_evidence_ids or known_evidence_ids
    for record in existing_records:
        evidence_id = str(record.get("evidenceId", ""))
        should_synchronize = evidence_id in selected_or_known_ids
        remains_approved = evidence_id in approved_ids_for_sync
        current_registry = record.get("registryFingerprint") == fingerprint
        if should_synchronize and (not remains_approved or not current_registry):
            continue
        validate_training_record(record, valid_route_ids, str(args.output))
        retained_records.append(record)

    reference_index: dict[str, tuple[str, tuple[str, ...]]] = {}
    for reference_path in args.reference_data:
        for record in read_jsonl(reference_path):
            validate_training_record(record, valid_route_ids, str(reference_path))
            normalized = normalize_query(record["query"])
            label = label_key(record["scope"], record["relevantRouteIds"])
            previous_label = reference_index.get(normalized)
            if previous_label and previous_label != label:
                raise ValueError(
                    f"Existing reference data assigns conflicting labels to query {record['query']!r}."
                )
            reference_index[normalized] = label

    created_at = datetime.now(timezone.utc).isoformat()
    generated_records = list(retained_records)
    report = {
        "schemaVersion": 1,
        "generatedAt": created_at,
        "generatorModel": args.model,
        "validatorModel": None if args.skip_semantic_validation else args.validator_model,
        "ollamaUrl": args.ollama_url,
        "registryFingerprint": fingerprint,
        "approvedCorrectionsFound": len(all_approved_corrections),
        "duplicateApprovedCorrectionsSkipped": duplicate_correction_count,
        "correctionsProcessed": 0,
        "originalQueriesAdded": 0,
        "paraphrasesAdded": 0,
        "deterministicallyRejected": 0,
        "nearDuplicatesRejected": 0,
        "semanticValidationRejected": 0,
        "duplicateTrainingQueriesRejected": 0,
        "excessExistingParaphrasesRemoved": 0,
        "corrections": [],
    }

    for correction_index, correction in enumerate(corrections):
        evidence_id = correction["evidenceId"]
        expected_label = label_key(correction["scope"], correction["relevantRouteIds"])
        evidence_records = [
            record for record in generated_records if str(record.get("evidenceId", "")) == evidence_id
        ]
        labels_are_current = all(
            label_key(record["scope"], record["relevantRouteIds"]) == expected_label
            for record in evidence_records
        )
        if not labels_are_current:
            generated_records = [
                record
                for record in generated_records
                if str(record.get("evidenceId", "")) != evidence_id
            ]
            evidence_records = []

        existing_original = any(
            record.get("source") == "approved-interaction-correction"
            for record in evidence_records
        )
        if not existing_original:
            original_normalized = normalize_query(correction["query"])
            reference_label = reference_index.get(original_normalized)
            if reference_label and reference_label != expected_label:
                raise ValueError(
                    f"Approved correction {evidence_id} conflicts with existing training data "
                    f"for query {correction['query']!r}."
                )
            generated_records.append(
                {
                    "query": correction["query"],
                    "scope": correction["scope"],
                    "relevantRouteIds": correction["relevantRouteIds"],
                    "source": "approved-interaction-correction",
                    "evidenceId": evidence_id,
                    "registryFingerprint": fingerprint,
                    "reviewedAt": correction["reviewedAt"],
                }
            )
            report["originalQueriesAdded"] += 1

        existing_paraphrases = [
            record
            for record in evidence_records
            if record.get("source") == "synthetic-hard-example"
        ]
        if len(existing_paraphrases) > args.max_paraphrases:
            excess_records = existing_paraphrases[args.max_paraphrases :]
            excess_record_ids = {id(record) for record in excess_records}
            generated_records = [
                record for record in generated_records if id(record) not in excess_record_ids
            ]
            existing_paraphrases = existing_paraphrases[: args.max_paraphrases]
            report["excessExistingParaphrasesRemoved"] += len(excess_records)
        remaining_slots = max(0, args.max_paraphrases - len(existing_paraphrases))
        correction_report = {
            "evidenceId": evidence_id,
            "scope": correction["scope"],
            "routeIds": correction["relevantRouteIds"],
            "existingParaphrases": len(existing_paraphrases),
            "remainingSlots": remaining_slots,
            "acceptedParaphrases": 0,
        }
        report["correctionsProcessed"] += 1

        if args.dry_run or remaining_slots == 0:
            report["corrections"].append(correction_report)
            continue

        approved_route_ids = set(correction["relevantRouteIds"])
        routing_context = {
            "approvedRoutes": [
                compact_route(routes_by_id[route_id])
                for route_id in correction["relevantRouteIds"]
            ],
            "rejectedSuggestedRoutes": [
                compact_route(routes_by_id[route_id])
                for route_id in correction["suggestedRouteIds"]
                if route_id not in approved_route_ids and route_id in routes_by_id
            ],
        }
        generated_candidates, generation_metrics = generate_candidates(
            args,
            correction,
            routing_context,
            args.seed + correction_index,
        )
        deterministic_candidates = []
        for candidate in generated_candidates:
            reason = deterministic_rejection_reason(candidate, correction, valid_route_ids)
            if reason:
                report["deterministicallyRejected"] += 1
                continue
            deterministic_candidates.append(candidate)

        existing_query_texts = [
            record["query"]
            for record in generated_records
            if label_key(record["scope"], record["relevantRouteIds"]) == expected_label
        ]
        deterministic_candidates, near_duplicate_count = remove_near_duplicates(
            deterministic_candidates,
            existing_query_texts,
        )
        report["nearDuplicatesRejected"] += near_duplicate_count
        semantically_accepted, validation_metrics = semantic_validation(
            args,
            correction,
            routing_context,
            deterministic_candidates,
            args.seed + correction_index,
        )
        report["semanticValidationRejected"] += (
            len(deterministic_candidates) - len(semantically_accepted)
        )

        accepted_for_correction = []
        current_output_queries = {
            normalize_query(record["query"]): label_key(
                record["scope"], record["relevantRouteIds"]
            )
            for record in generated_records
        }
        for candidate in semantically_accepted:
            normalized = normalize_query(candidate)
            known_label = reference_index.get(normalized) or current_output_queries.get(normalized)
            if known_label:
                report["duplicateTrainingQueriesRejected"] += 1
                continue
            accepted_for_correction.append(candidate)
            current_output_queries[normalized] = expected_label
            if len(accepted_for_correction) >= remaining_slots:
                break

        for candidate in accepted_for_correction:
            generated_records.append(
                {
                    "query": candidate,
                    "scope": correction["scope"],
                    "relevantRouteIds": correction["relevantRouteIds"],
                    "source": "synthetic-hard-example",
                    "evidenceId": evidence_id,
                    "generatorModel": args.model,
                    "semanticValidatorModel": (
                        None if args.skip_semantic_validation else args.validator_model
                    ),
                    "registryFingerprint": fingerprint,
                    "generatedAt": created_at,
                }
            )
        report["paraphrasesAdded"] += len(accepted_for_correction)
        correction_report.update(
            {
                "generatedCandidates": len(generated_candidates),
                "afterDeterministicChecks": len(deterministic_candidates),
                "afterSemanticValidation": len(semantically_accepted),
                "acceptedParaphrases": len(accepted_for_correction),
                "generationMetrics": generation_metrics,
                "validationMetrics": validation_metrics,
            }
        )
        report["corrections"].append(correction_report)

    generated_records.sort(
        key=lambda record: (
            str(record.get("evidenceId", "")),
            0 if record.get("source") == "approved-interaction-correction" else 1,
            normalize_query(record["query"]),
        )
    )
    report["totalOutputExamples"] = len(generated_records)

    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        print("Dry run complete; Ollama was not called and no files were written.")
        return

    write_jsonl_atomic(args.output, generated_records)
    write_json_atomic(args.manifest, report)
    print(
        f"Hard examples: corrections={report['correctionsProcessed']} "
        f"originalsAdded={report['originalQueriesAdded']} "
        f"paraphrasesAdded={report['paraphrasesAdded']} "
        f"total={report['totalOutputExamples']}"
    )
    print(f"Training data: {args.output.resolve()}")
    print(f"Generation manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
