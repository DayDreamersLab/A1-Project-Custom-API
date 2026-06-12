from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import json
import sys

from pytorch_route_ranker.scripts import generate_hard_examples


class GenerateHardExamplesTest(TestCase):
    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(f"{json.dumps(record)}\n" for record in records),
            encoding="utf-8",
        )

    def test_deterministic_checks_preserve_scope_and_current_qualifier(self):
        correction = {
            "query": "show current runway wind",
            "scope": "single",
            "relevantRouteIds": ["wind-runway-impact"],
        }
        route_ids = {"wind-runway-impact"}

        self.assertIsNone(
            generate_hard_examples.deterministic_rejection_reason(
                "latest wind affecting the runway",
                correction,
                route_ids,
            )
        )
        self.assertEqual(
            generate_hard_examples.deterministic_rejection_reason(
                "show all current runway wind information",
                correction,
                route_ids,
            ),
            "changed-single-query-to-multiple",
        )
        self.assertEqual(
            generate_hard_examples.deterministic_rejection_reason(
                "show runway wind information",
                correction,
                route_ids,
            ),
            "lost-current-qualifier",
        )

    def test_generation_and_validation_use_their_configured_models(self):
        args = SimpleNamespace(
            generate_count=3,
            ollama_url="http://127.0.0.1:11434/api/chat",
            model="generator-model",
            validator_model="validator-model",
            timeout_seconds=10,
            retries=0,
            skip_semantic_validation=False,
        )
        correction = {
            "query": "show current runway wind",
            "scope": "single",
            "relevantRouteIds": ["wind-runway-impact"],
        }
        routing_context = {"approvedRoutes": [], "rejectedSuggestedRoutes": []}

        with patch.object(
            generate_hard_examples,
            "call_ollama",
            side_effect=[
                ({"paraphrases": ["latest runway wind"]}, {}),
                ({"acceptedIndexes": [0]}, {}),
            ],
        ) as ollama_call:
            candidates, _ = generate_hard_examples.generate_candidates(
                args,
                correction,
                routing_context,
                42,
            )
            generate_hard_examples.semantic_validation(
                args,
                correction,
                routing_context,
                candidates,
                42,
            )

        self.assertEqual(ollama_call.call_args_list[0].kwargs["model"], "generator-model")
        self.assertEqual(ollama_call.call_args_list[1].kwargs["model"], "validator-model")

    def test_pipeline_uses_approved_labels_and_removes_revoked_evidence(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry_path = root / "route_registry.json"
            evidence_path = root / "evidence.jsonl"
            output_path = root / "hard.jsonl"
            manifest_path = root / "manifest.json"
            registry_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "wind-runway-impact",
                            "title": "Runway Wind Impact",
                            "path": "#",
                            "description": "Current runway-specific wind impact.",
                            "keywords": ["runway wind", "crosswind"],
                        },
                        {
                            "id": "wind-current-observations",
                            "title": "Current Wind Observations",
                            "path": "#",
                            "description": "Current general wind observations.",
                            "keywords": ["current wind", "observations"],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            approved_evidence = {
                "evidenceId": "correction-1",
                "query": "show current runway wind",
                "suggestedRouteIds": [
                    "wind-runway-impact",
                    "wind-current-observations",
                ],
                "reviewStatus": "approved",
                "approvedScope": "single",
                "approvedRelevantRouteIds": ["wind-runway-impact"],
            }
            self.write_jsonl(evidence_path, [approved_evidence])
            arguments = [
                "generate_hard_examples.py",
                "--registry",
                str(registry_path),
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
                "--manifest",
                str(manifest_path),
                "--reference-data",
            ]

            with (
                patch.object(
                    generate_hard_examples,
                    "generate_candidates",
                    return_value=(
                        [
                            "latest wind affecting the runway",
                            "show all current runway wind information",
                            "show runway wind information",
                        ],
                        {"roundTripMs": 1},
                    ),
                ),
                patch.object(
                    generate_hard_examples,
                    "semantic_validation",
                    side_effect=lambda _args, _correction, _routes, candidates, _seed: (
                        candidates,
                        {"roundTripMs": 1},
                    ),
                ),
                patch.object(sys, "argv", arguments),
            ):
                generate_hard_examples.main()

            records = generate_hard_examples.read_jsonl(output_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(
                {record["query"] for record in records},
                {"show current runway wind", "latest wind affecting the runway"},
            )
            self.assertTrue(
                all(record["relevantRouteIds"] == ["wind-runway-impact"] for record in records)
            )

            revoked_evidence = {**approved_evidence, "reviewStatus": "rejected"}
            self.write_jsonl(evidence_path, [revoked_evidence])
            with patch.object(sys, "argv", arguments):
                generate_hard_examples.main()

            self.assertEqual(generate_hard_examples.read_jsonl(output_path), [])


if __name__ == "__main__":
    import unittest

    unittest.main()
