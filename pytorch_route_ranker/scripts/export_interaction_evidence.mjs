import { readFile } from "node:fs/promises";

import { routeRegistry } from "../../src/data/routeRegistry.js";
import {
  defaultReviewableEvidenceUrl,
  exportReviewableInteractionEvidence,
} from "../../server/interactionEvidenceExporter.mjs";

const personalizationStoreUrl = new URL(
  "../../server/data/personalizationStore.json",
  import.meta.url
);

let selectionEvidence = [];

try {
  const store = JSON.parse(await readFile(personalizationStoreUrl, "utf8"));
  selectionEvidence = Array.isArray(store.selectionEvidence) ? store.selectionEvidence : [];
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const result = await exportReviewableInteractionEvidence({
  selectionEvidence,
  routeRegistry,
  output: defaultReviewableEvidenceUrl,
});

console.log(
  `Exported review queue: added=${result.added} updated=${result.updated} total=${result.total}`
);
console.log(`Reviewable interaction evidence: ${result.outputPath}`);
