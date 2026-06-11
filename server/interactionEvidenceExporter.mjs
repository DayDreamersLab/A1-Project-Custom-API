import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

export const defaultReviewableEvidenceUrl = new URL(
  "../pytorch_route_ranker/data/reviewable_interaction_evidence.jsonl",
  import.meta.url
);

const preservedReviewFields = [
  "reviewStatus",
  "reviewNotes",
  "reviewedAt",
  "reviewedBy",
  "approvedScope",
  "approvedRelevantRouteIds",
];

function uniqueStrings(values) {
  return [...new Set((Array.isArray(values) ? values : []).filter(Boolean).map(String))];
}

function outputPathFrom(value) {
  return value instanceof URL ? fileURLToPath(value) : String(value);
}

function preserveReview(existingRecord) {
  return Object.fromEntries(
    preservedReviewFields
      .filter((field) => existingRecord?.[field] !== undefined)
      .map((field) => [field, existingRecord[field]])
  );
}

function validateRouteIds(routeIds, approvedRouteIds, fieldName, evidenceId) {
  const invalidRouteIds = routeIds.filter((routeId) => !approvedRouteIds.has(routeId));
  if (invalidRouteIds.length > 0) {
    throw new Error(
      `Selection evidence ${evidenceId} contains unapproved ${fieldName}: ${invalidRouteIds.join(", ")}`
    );
  }
}

export function createReviewableEvidenceRecord(selectionEvidence, routeRegistry) {
  const approvedRouteIds = new Set(routeRegistry.map((route) => route.id));
  const evidenceId = String(
    selectionEvidence.id ??
      `${selectionEvidence.recommendationId ?? "unknown-recommendation"}:${selectionEvidence.timestamp ?? "unknown-time"}`
  );
  const query = String(selectionEvidence.query ?? "").trim();
  const outcome = selectionEvidence.outcome === "none-match" ? "none-match" : "selected";
  const selectedRouteIds = uniqueStrings(selectionEvidence.selectedRouteIds);
  const suggestedRouteIds = uniqueStrings(selectionEvidence.suggestedRouteIds);

  if (!query) {
    throw new Error(`Selection evidence ${evidenceId} has no reviewable query.`);
  }
  if (suggestedRouteIds.length === 0) {
    throw new Error(`Selection evidence ${evidenceId} has no suggested routes.`);
  }

  validateRouteIds(selectedRouteIds, approvedRouteIds, "selected route IDs", evidenceId);
  validateRouteIds(suggestedRouteIds, approvedRouteIds, "suggested route IDs", evidenceId);

  if (selectedRouteIds.some((routeId) => !suggestedRouteIds.includes(routeId))) {
    throw new Error(`Selection evidence ${evidenceId} selected a route that was not suggested.`);
  }
  if (outcome === "selected" && selectedRouteIds.length === 0) {
    throw new Error(`Selection evidence ${evidenceId} has a selected outcome without routes.`);
  }
  if (outcome === "none-match" && selectedRouteIds.length > 0) {
    throw new Error(`Selection evidence ${evidenceId} is none-match but contains selected routes.`);
  }

  return {
    schemaVersion: 1,
    evidenceId,
    recommendationId: selectionEvidence.recommendationId ?? null,
    timestamp: selectionEvidence.timestamp ?? null,
    query,
    roleKey: selectionEvidence.roleKey ?? "unknown-role",
    outcome,
    suggestedRouteIds,
    proposedScope:
      outcome === "selected" ? (selectedRouteIds.length > 1 ? "multiple" : "single") : null,
    proposedRelevantRouteIds: selectedRouteIds,
    reviewStatus: "pending",
    reviewNotes: "",
    reviewedAt: null,
    reviewedBy: null,
    approvedScope: null,
    approvedRelevantRouteIds: [],
    reviewReason:
      outcome === "selected"
        ? "verify-user-selected-routes"
        : "assign-correct-routes-after-none-match",
    source: "interaction-clarification",
  };
}

async function readExistingRecords(outputPath) {
  try {
    const contents = await readFile(outputPath, "utf8");
    return contents
      .split(/\r?\n/)
      .filter((line) => line.trim())
      .map((line, index) => {
        try {
          return JSON.parse(line);
        } catch (error) {
          throw new Error(
            `Could not parse ${outputPath} line ${index + 1}: ${error.message}`
          );
        }
      });
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
}

export async function exportReviewableInteractionEvidence({
  selectionEvidence,
  routeRegistry,
  output = defaultReviewableEvidenceUrl,
}) {
  const outputPath = outputPathFrom(output);
  const existingRecords = await readExistingRecords(outputPath);
  const recordsById = new Map(
    existingRecords
      .filter((record) => record?.evidenceId)
      .map((record) => [String(record.evidenceId), record])
  );
  let added = 0;
  let updated = 0;

  for (const evidence of selectionEvidence ?? []) {
    const generatedRecord = createReviewableEvidenceRecord(evidence, routeRegistry);
    const existingRecord = recordsById.get(generatedRecord.evidenceId);
    recordsById.set(generatedRecord.evidenceId, {
      ...generatedRecord,
      ...preserveReview(existingRecord),
    });
    if (existingRecord) {
      updated += 1;
    } else {
      added += 1;
    }
  }

  const records = [...recordsById.values()].sort((recordA, recordB) =>
    String(recordA.timestamp ?? "").localeCompare(String(recordB.timestamp ?? ""))
  );
  const jsonl = records.map((record) => JSON.stringify(record)).join("\n");
  const temporaryPath = `${outputPath}.tmp-${process.pid}`;

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(temporaryPath, jsonl ? `${jsonl}\n` : "", "utf8");
  await rename(temporaryPath, outputPath);

  return {
    outputPath,
    added,
    updated,
    total: records.length,
  };
}
