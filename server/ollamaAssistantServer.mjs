// Imports Node's HTTP server API.
import http from "node:http";
// Imports hashing and UUID helpers from Node.
import { createHash, randomUUID } from "node:crypto";
// Imports asynchronous filesystem helpers from Node.
import { mkdir, readFile, writeFile } from "node:fs/promises";
// Imports the approved AMIDS route registry.
import { routeRegistry } from "../src/data/routeRegistry.js";
// Begins importing shared query-intent helpers.
import {
  // Imports the query-normalization helper.
  normalize,
  // Imports the helper that detects explicit all/every requests.
  queryRequestsAllRelated,
  // Imports the helper that detects navigation intent.
  queryRequestsOpen,
  // Imports the helper that splits queries into tokens.
  tokenize,
// Finishes importing shared query-intent helpers.
} from "../src/services/queryIntent.js";

// Reads the local assistant API port from the environment, defaulting to 3001.
const PORT = Number(process.env.AMIDS_ASSISTANT_PORT ?? 3001);
// Reads the Ollama chat endpoint, defaulting to the local Ollama service.
const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://127.0.0.1:11434/api/chat";
// Selects the configured Ollama routing model, with Qwen 3 as the default.
const OLLAMA_ROUTING_MODEL =
  // Continues the current operation.
  process.env.OLLAMA_ROUTING_MODEL ?? process.env.OLLAMA_MODEL ?? "qwen3:latest";
// Selects Ollama-only, PyTorch-only, or PyTorch-first routing.
const routingProvider = ["ollama", "pytorch", "hybrid"].includes(
  String(process.env.AMIDS_ROUTING_PROVIDER ?? "hybrid").toLowerCase()
)
  ? String(process.env.AMIDS_ROUTING_PROVIDER ?? "hybrid").toLowerCase()
  : "hybrid";
// Points to the local specialised PyTorch route-ranking endpoint.
const PYTORCH_RANKER_URL =
  process.env.AMIDS_PYTORCH_RANKER_URL ?? "http://127.0.0.1:8001/rank";
// Prevents an unavailable ranker from delaying fallback for too long.
const pytorchRankerTimeoutMs = Math.max(
  100,
  Number(process.env.AMIDS_PYTORCH_RANKER_TIMEOUT_MS ?? 2_000)
);
// Limits how many related routes a multiple-route response can contain.
const maxRelatedRoutes = 8;
// Limits how many retrieved routes are sent to the AI model.
const qwenCandidateLimit = Math.max(1, Number(process.env.AMIDS_QWEN_CANDIDATE_LIMIT ?? 12));
// Gives broad requests a larger pool so the model can select several coherent routes.
const broadQwenCandidateLimit = Math.max(
  qwenCandidateLimit,
  Number(process.env.AMIDS_BROAD_QWEN_CANDIDATE_LIMIT ?? 16)
);
// Changes whenever prompt behavior changes so outdated cached decisions are not reused.
const assistantPromptVersion = "explicit-broad-scope-v2";
// Controls how long Ollama keeps the selected model loaded.
const ollamaKeepAlive = process.env.OLLAMA_KEEP_ALIVE ?? -1;
// Sets the maximum Ollama context size used for routing.
const ollamaNumCtx = Number(process.env.OLLAMA_NUM_CTX ?? 2048);
// Provides enough context for the larger candidate pool used by broad requests.
const broadOllamaNumCtx = Math.max(
  ollamaNumCtx,
  Number(process.env.OLLAMA_BROAD_NUM_CTX ?? 3072)
);
// Limits how many output tokens Ollama may generate.
const ollamaNumPredict = Number(process.env.OLLAMA_NUM_PREDICT ?? 160);
// Sets how long a cached assistant result remains valid.
const assistantCacheTtlMs = Math.max(
  // Adds this value to the current structure.
  0,
  // Continues the current operation.
  Number(process.env.AMIDS_ASSISTANT_CACHE_TTL_MS ?? 300_000)
);
// Limits the number of assistant results held in memory.
const assistantCacheMaxEntries = Math.max(
  // Adds this value to the current structure.
  1,
  // Continues the current operation.
  Number(process.env.AMIDS_ASSISTANT_CACHE_MAX_ENTRIES ?? 200)
);
// Defines the fallback prototype user identifier.
const defaultUserId = "prototype-user";
// Points to the directory containing local personalization data.
const storeDirectoryUrl = new URL("./data/", import.meta.url);
// Points to the JSON file used for personalization storage.
const storeFileUrl = new URL("./data/personalizationStore.json", import.meta.url);
// Stores recently validated assistant responses in memory.
const assistantResponseCache = new Map();
// Tracks active identical requests so they can share one model call.
const assistantInFlightRequests = new Map();
// Serializes feedback updates to prevent concurrent duplicate scoring.
let feedbackProcessingQueue = Promise.resolve();
// Creates a registry fingerprint used when building cache keys.
const routeRegistryVersion = createHash("sha256")
  // Adds normalized data to the hash input.
  .update(JSON.stringify(routeRegistry))
  // Converts the completed hash to hexadecimal text.
  .digest("hex")
  // Limits the current value to the required size.
  .slice(0, 12);

// Defines response and CORS headers for JSON API responses.
const jsonHeaders = {
  // Adds an instruction or value to the current structure.
  "Access-Control-Allow-Headers": "Content-Type",
  // Adds an instruction or value to the current structure.
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  // Adds an instruction or value to the current structure.
  "Access-Control-Allow-Origin": process.env.ALLOWED_ORIGIN ?? "http://localhost:5173",
  // Adds an instruction or value to the current structure.
  "Content-Type": "application/json",
};

// Defines response and CORS headers for simulated HTML route pages.
const htmlHeaders = {
  // Adds an instruction or value to the current structure.
  "Access-Control-Allow-Origin": process.env.ALLOWED_ORIGIN ?? "http://localhost:5173",
  // Adds an instruction or value to the current structure.
  "Content-Type": "text/html; charset=utf-8",
};

// Reads and parses a JSON body from an incoming HTTP request.
function readBody(request) {
  // Returns a promise for the asynchronous operation.
  return new Promise((resolve, reject) => {
    // Accumulates the incoming HTTP request body.
    let body = "";

    // Handles each incoming request-body chunk.
    request.on("data", (chunk) => {
      // Assigns the computed value for the current operation.
      body += chunk;
    });
    // Handles completion of the incoming request body.
    request.on("end", () => {
      // Starts an operation that may fail.
      try {
        // Continues the current operation.
        resolve(body ? JSON.parse(body) : {});
      // Handles an error from the preceding operation.
      } catch (error) {
        // Continues the current operation.
        reject(error);
      }
    });
    // Rejects body parsing when the request stream fails.
    request.on("error", reject);
  });
}

// Sends a JSON response with the shared API headers.
function sendJson(response, statusCode, payload) {
  // Writes the HTTP status and response headers.
  response.writeHead(statusCode, jsonHeaders);
  // Finishes the HTTP response.
  response.end(JSON.stringify(payload));
}

// Sends an HTML response with the shared HTML headers.
function sendHtml(response, statusCode, html) {
  // Writes the HTTP status and response headers.
  response.writeHead(statusCode, htmlHeaders);
  // Finishes the HTTP response.
  response.end(html);
}

// Escapes untrusted text before inserting it into generated HTML.
function escapeHtml(value) {
  // Converts the value to text before processing it.
  return String(value)
    // Escapes this HTML-sensitive character.
    .replaceAll("&", "&amp;")
    // Escapes this HTML-sensitive character.
    .replaceAll("<", "&lt;")
    // Escapes this HTML-sensitive character.
    .replaceAll(">", "&gt;")
    // Escapes this HTML-sensitive character.
    .replaceAll('"', "&quot;")
    // Escapes this HTML-sensitive character.
    .replaceAll("'", "&#039;");
}

// Builds the simulated page for an approved routeRegistry entry.
function renderRoutePage(route) {
  // Returns one complete HTML document; its internal HTML and CSS are kept uncommented so comments never enter the generated page.
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>${escapeHtml(route.title)}</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #eef3f8;
        --primary: #1769aa;
        --text: #142033;
        --muted: #5d6b7e;
        --line: #d8e1ea;
      }

      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background: var(--bg);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      main {
        width: min(780px, calc(100% - 32px));
        padding: 28px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 18px 50px rgba(20, 32, 51, 0.12);
      }

      p {
        color: var(--muted);
        line-height: 1.6;
      }

      code {
        color: var(--primary);
        font-weight: 800;
      }
    </style>
  </head>
  <body>
    <main>
      <p>AMIDS Route Registry Target</p>
      <h1>${escapeHtml(route.title)}</h1>
      <p>${escapeHtml(route.description)}</p>
      <p>Route ID: <code>${escapeHtml(route.id)}</code></p>
      <p>Registry path: <code>${escapeHtml(route.path)}</code></p>
      <p>
        In the real AMIDS environment, replace this registry path with the
        corresponding internal AMIDS URL in <code>routeRegistry</code>.
      </p>
    </main>
  </body>
</html>`;
}

// Removes falsy values and duplicates from a list.
function unique(values) {
  // Returns a newly constructed list.
  return [...new Set(values.filter(Boolean))];
}

// Returns only route fields safe for the frontend.
function toPublicRoute(route) {
  // Returns a structured result object.
  return {
    // Sets this property in the current object.
    id: route.id,
    // Sets this property in the current object.
    title: route.title,
    // Sets this property in the current object.
    path: route.path,
    // Sets this property in the current object.
    description: route.description,
  };
}

// Creates a stable hash for one query-and-routes recommendation.
function createRecommendationId(payload, routeIds) {
  // Returns a deterministic SHA-256 hash.
  return createHash("sha256")
    // Adds normalized data to the hash input.
    .update(
      // Opens the next processing block.
      JSON.stringify({
        // Sets this property in the current object.
        query: normalize(payload.query),
        // Sets this property in the current object.
        roleKey: payload.roleKey ?? "unknown-role",
        // Sets this property in the current object.
        routeIds: unique(routeIds).sort(),
      })
    // Continues the current operation.
    )
    // Converts the completed hash to hexadecimal text.
    .digest("hex")
    // Limits the current value to the required size.
    .slice(0, 24);
}

// Creates an empty personalization-store structure.
function createEmptyStore() {
  // Returns a structured result object.
  return {
    // Sets this property in the current object.
    profiles: {},
    // Sets this property in the current object.
    feedback: [],
  };
}

// Builds the storage key for one user-role profile.
function profileKey(userId, roleKey) {
  // Returns the computed result to the caller.
  return `${userId || defaultUserId}::${roleKey || "unknown-role"}`;
}

// Creates a new empty personalization profile.
function createDefaultProfile(userId, roleKey) {
  // Captures the current time in ISO format.
  const timestamp = new Date().toISOString();

  // Returns a structured result object.
  return {
    // Adds this value to the current structure.
    userId,
    // Adds this value to the current structure.
    roleKey,
    // Sets this property in the current object.
    preferredRouteIds: [],
    // Sets this property in the current object.
    avoidedRouteIds: [],
    // Sets this property in the current object.
    frequentTopics: [],
    // Sets this property in the current object.
    routeScores: {},
    // Sets this property in the current object.
    topicScores: {},
    // Sets this property in the current object.
    recentQueries: [],
    // Sets this property in the current object.
    feedbackCount: 0,
    // Sets this property in the current object.
    createdAt: timestamp,
    // Sets this property in the current object.
    updatedAt: timestamp,
  };

}

// Loads personalization data from disk or returns an empty store.
async function readPersonalizationStore() {
  // Starts an operation that may fail.
  try {
    // Reads the saved personalization JSON as text.
    const contents = await readFile(storeFileUrl, "utf8");
    // Parses the saved personalization JSON into an object.
    const parsedStore = JSON.parse(contents);
    // Returns a structured result object.
    return {
      // Spreads these values into the current structure.
      ...createEmptyStore(),
      // Spreads these values into the current structure.
      ...parsedStore,
      // Sets this property in the current object.
      profiles: parsedStore.profiles ?? {},
      // Sets this property in the current object.
      feedback: parsedStore.feedback ?? [],
    };
  // Handles an error from the preceding operation.
  } catch {
    // Returns the computed result to the caller.
    return createEmptyStore();
  }

}

// Writes personalization data to disk.
async function writePersonalizationStore(store) {
  // Ensures the personalization directory exists.
  await mkdir(storeDirectoryUrl, { recursive: true });
  // Persists the updated JSON store to disk.
  await writeFile(storeFileUrl, JSON.stringify(store, null, 2), "utf8");
}

// Gets an existing profile or creates one when missing.
function ensureProfile(store, userId = defaultUserId, roleKey = "unknown-role") {
  // Builds the unique storage key for a user and role.
  const key = profileKey(userId, roleKey);

  // Checks whether this condition is true.
  if (!store.profiles[key]) {
    // Assigns the computed value for the current operation.
    store.profiles[key] = createDefaultProfile(userId, roleKey);
  }

  // Returns the computed result to the caller.
  return store.profiles[key];
}

// Rebuilds preferred, avoided, and frequent-topic summary lists.
function updateProfileLists(profile) {
  // Converts saved route scores into sortable key-value entries.
  const routeScoreEntries = Object.entries(profile.routeScores ?? {});
  // Assigns the computed value for the current operation.
  profile.preferredRouteIds = routeScoreEntries
    // Keeps only entries that satisfy this condition.
    .filter(([, score]) => score > 0)
    // Sorts entries into the required order.
    .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
    // Limits the current value to the required size.
    .slice(0, 12)
    // Transforms each entry into the required shape.
    .map(([routeId]) => routeId);
  // Assigns the computed value for the current operation.
  profile.avoidedRouteIds = routeScoreEntries
    // Keeps only entries that satisfy this condition.
    .filter(([, score]) => score < 0)
    // Sorts entries into the required order.
    .sort(([, scoreA], [, scoreB]) => scoreA - scoreB)
    // Limits the current value to the required size.
    .slice(0, 12)
    // Transforms each entry into the required shape.
    .map(([routeId]) => routeId);
  // Assigns the computed value for the current operation.
  profile.frequentTopics = Object.entries(profile.topicScores ?? {})
    // Keeps only entries that satisfy this condition.
    .filter(([, score]) => score > 0)
    // Sorts entries into the required order.
    .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
    // Limits the current value to the required size.
    .slice(0, 12)
    // Transforms each entry into the required shape.
    .map(([topic]) => topic);
}

// Collects all route IDs referenced by a feedback payload.
function getRouteIdsFromFeedback(payload) {
  // Returns a deduplicated list.
  return unique([
    // Adds this value to the current structure.
    payload.recommendedRouteId,
    // Spreads these values into the current structure.
    ...(payload.recommendedRouteIds ?? []),
    // Adds this value to the current structure.
    payload.result?.route?.id,
    // Spreads these values into the current structure.
    ...(payload.result?.routes?.map((route) => route.id) ?? []),
  // Continues the current operation.
  ]);
}

// Infers useful personalization topics from feedback.
function inferFeedbackTopics(query, routeIds) {
  // Lists generic query words that should not become personalization topics.
  const stopWords = new Set([
    // Adds an instruction or value to the current structure.
    "all",
    // Adds an instruction or value to the current structure.
    "and",
    // Adds an instruction or value to the current structure.
    "are",
    // Adds an instruction or value to the current structure.
    "data",
    // Adds an instruction or value to the current structure.
    "for",
    // Adds an instruction or value to the current structure.
    "from",
    // Adds an instruction or value to the current structure.
    "get",
    // Adds an instruction or value to the current structure.
    "give",
    // Adds an instruction or value to the current structure.
    "info",
    // Adds an instruction or value to the current structure.
    "information",
    // Adds an instruction or value to the current structure.
    "me",
    // Adds an instruction or value to the current structure.
    "need",
    // Adds an instruction or value to the current structure.
    "open",
    // Adds an instruction or value to the current structure.
    "related",
    // Adds an instruction or value to the current structure.
    "show",
    // Adds an instruction or value to the current structure.
    "the",
    // Adds an instruction or value to the current structure.
    "to",
    // Adds an instruction or value to the current structure.
    "with",
  // Continues the current operation.
  ]);
  // Extracts useful topic words from the user's query.
  const queryTopics = tokenize(query ?? "")
    // Transforms each entry into the required shape.
    .map((token) => (token.endsWith("s") ? token.slice(0, -1) : token))
    // Keeps only entries that satisfy this condition.
    .filter((token) => token.length > 2 && !stopWords.has(token));
  // Extracts broad topic prefixes from recommended route IDs.
  const routeTopics = routeIds.map((routeId) => routeId.split("-")[0]);

  // Returns a deduplicated list.
  return unique([...routeTopics, ...queryTopics]).slice(0, 12);
}

// Loads one user's personalization profile.
async function getPersonalizationProfile(userId = defaultUserId, roleKey = "unknown-role") {
  // Loads the current personalization store.
  const store = await readPersonalizationStore();
  // Gets or creates the personalization profile for this user and role.
  const profile = ensureProfile(store, userId, roleKey);
  // Continues the current operation.
  await writePersonalizationStore(store);
  // Returns the computed result to the caller.
  return profile;
}

// Applies one deduplicated feedback decision to personalization.
async function processFeedbackTransaction(payload) {
  // Loads the current personalization store.
  const store = await readPersonalizationStore();
  // Captures the current time in ISO format.
  const timestamp = new Date().toISOString();
  // Uses the supplied user ID or the prototype fallback ID.
  const userId = payload.userId || defaultUserId;
  // Uses the supplied role key or an unknown-role fallback.
  const roleKey = payload.roleKey || "unknown-role";
  // Normalizes feedback into helpful or not-helpful.
  const rating = payload.rating === "not-helpful" ? "not-helpful" : "helpful";
  // Collects the route IDs associated with the recommendation.
  const routeIds = getRouteIdsFromFeedback(payload);
  // Creates a stable identity for this recommendation.
  const recommendationId = createRecommendationId(payload, routeIds);
  // Identifies the primary recommended route.
  const selectedRouteId = payload.recommendedRouteId ?? routeIds[0] ?? null;
  // Gets or creates the personalization profile for this user and role.
  const profile = ensureProfile(store, userId, roleKey);
  // Searches for feedback already submitted for this recommendation.
  const existingFeedback = (store.feedback ?? []).find(
    // Assigns the computed value for the current operation.
    (record) =>
      // Assigns the computed value for the current operation.
      record.userId === userId &&
      // Assigns the computed value for the current operation.
      record.roleKey === roleKey &&
      // Assigns the computed value for the current operation.
      record.recommendationId === recommendationId
  );

  // Stops before changing scores when this recommendation already has feedback.
  if (existingFeedback) {
    // Returns a structured result object.
    return {
      // Sets this property in the current object.
      ok: true,
      // Sets this property in the current object.
      duplicate: true,
      // Sets this property in the current object.
      mode: "profile-update-duplicate",
      // Sets this property in the current object.
      feedbackRecord: existingFeedback,
      // Adds this value to the current structure.
      profile,
      // Sets this property in the current object.
      profileDelta: null,
    };
  }

  // Sets the personalization score change for the primary route.
  const routeDelta = rating === "helpful" ? 2 : -2;
  // Sets the score change for additional recommended routes.
  const secondaryRouteDelta = rating === "helpful" ? 1 : -1;

  // Processes each entry in the current list.
  routeIds.forEach((routeId, index) => {
    // Selects the score adjustment for the current route.
    const delta = index === 0 ? routeDelta : secondaryRouteDelta;
    // Assigns the computed value for the current operation.
    profile.routeScores[routeId] = (profile.routeScores[routeId] ?? 0) + delta;
  });

  // Processes each entry in the current list.
  inferFeedbackTopics(payload.query, routeIds).forEach((topic) => {
    // Assigns the computed value for the current operation.
    profile.topicScores[topic] = (profile.topicScores[topic] ?? 0) + 1;
  });

  // Assigns the computed value for the current operation.
  profile.recentQueries = [
    // Opens the next processing block.
    {
      // Sets this property in the current object.
      query: payload.query,
      // Adds this value to the current structure.
      rating,
      // Adds this value to the current structure.
      routeIds,
      // Adds this value to the current structure.
      timestamp,
    },
    // Spreads these values into the current structure.
    ...(profile.recentQueries ?? []),
  // Continues the current operation.
  ].slice(0, 12);
  // Assigns the computed value for the current operation.
  profile.feedbackCount = (profile.feedbackCount ?? 0) + 1;
  // Assigns the computed value for the current operation.
  profile.updatedAt = timestamp;
  // Continues the current operation.
  updateProfileLists(profile);

  // Builds the feedback record saved for auditing and personalization.
  const feedbackRecord = {
    // Sets this property in the current object.
    id: payload.id ?? randomUUID(),
    // Adds this value to the current structure.
    userId,
    // Adds this value to the current structure.
    roleKey,
    // Adds this value to the current structure.
    recommendationId,
    // Sets this property in the current object.
    query: payload.query,
    // Adds this value to the current structure.
    rating,
    // Adds this value to the current structure.
    selectedRouteId,
    // Sets this property in the current object.
    recommendedRouteIds: routeIds,
    // Adds this value to the current structure.
    timestamp,
  };

  // Assigns the computed value for the current operation.
  store.feedback = [feedbackRecord, ...(store.feedback ?? [])].slice(0, 250);
  // Continues the current operation.
  await writePersonalizationStore(store);
  // Clears cached results after personalization changes.
  assistantResponseCache.clear();

  // Returns a structured result object.
  return {
    // Sets this property in the current object.
    ok: true,
    // Sets this property in the current object.
    mode: "profile-update-local",
    // Adds this value to the current structure.
    feedbackRecord,
    // Adds this value to the current structure.
    profile,
    // Sets this property in the current object.
    profileDelta: {
      // Adds this value to the current structure.
      routeIds,
      // Sets this property in the current object.
      topics: inferFeedbackTopics(payload.query, routeIds),
      // Adds this value to the current structure.
      rating,
    },
  };
}

// Queues feedback processing to avoid concurrent duplicate updates.
function processFeedback(payload) {
  // Queues this feedback update after earlier feedback operations.
  const operation = feedbackProcessingQueue.then(() => processFeedbackTransaction(payload));
  // Assigns the computed value for the current operation.
  feedbackProcessingQueue = operation.catch(() => undefined);
  // Returns the computed result to the caller.
  return operation;
}

// Calculates elapsed milliseconds from a start timestamp.
function elapsedMs(startedAt) {
  // Returns the computed result to the caller.
  return Date.now() - startedAt;
}

// Converts Ollama nanosecond metrics to milliseconds.
function nanosecondsToMs(value) {
  // Returns the computed result to the caller.
  return typeof value === "number" ? Number((value / 1_000_000).toFixed(2)) : null;
}

// Builds readable timing diagnostics from an Ollama response.
function summarizeOllamaMetrics(data, qwenStartedAt) {
  // Measures the complete local Ollama request duration.
  const roundTripDurationMs = elapsedMs(qwenStartedAt);
  // Converts Ollama model-loading time to milliseconds.
  const loadDurationMs = nanosecondsToMs(data.load_duration);
  // Converts Ollama prompt-processing time to milliseconds.
  const promptEvalDurationMs = nanosecondsToMs(data.prompt_eval_duration);
  // Converts Ollama output-generation time to milliseconds.
  const evalDurationMs = nanosecondsToMs(data.eval_duration);
  // Adds the model-reported inference timing components.
  const measuredInferenceMs = [loadDurationMs, promptEvalDurationMs, evalDurationMs]
    // Keeps only entries that satisfy this condition.
    .filter((value) => typeof value === "number")
    // Combines entries into one accumulated score or value.
    .reduce((total, value) => total + value, 0);

  // Returns a structured result object.
  return {
    // Adds this value to the current structure.
    roundTripDurationMs,
    // Sets this property in the current object.
    requestDurationMs: roundTripDurationMs,
    // Adds this value to the current structure.
    loadDurationMs,
    // Adds this value to the current structure.
    promptEvalDurationMs,
    // Adds this value to the current structure.
    evalDurationMs,
    // Sets this property in the current object.
    overheadDurationMs: Number(Math.max(0, roundTripDurationMs - measuredInferenceMs).toFixed(2)),
    // Sets this property in the current object.
    promptEvalCount: data.prompt_eval_count ?? null,
    // Sets this property in the current object.
    evalCount: data.eval_count ?? null,
  };
}

// Returns names of expert rules triggered by the query.
function getAppliedRules(payload) {
  // Returns the combined expression result.
  return (payload.expertRules ?? [])
    // Keeps only entries that satisfy this condition.
    .filter((rule) => rule.trigger && normalize(payload.query).includes(normalize(rule.trigger)))
    // Transforms each entry into the required shape.
    .map((rule) => rule.name);
}

// Summarizes the top retrieval scores.
function buildScoreSummary(scoredRoutes, matchedRoutes, scoringDurationMs) {
  // Gets the highest-scoring retrieved route.
  const topRoute = scoredRoutes[0] ?? null;
  // Gets the second-highest-scoring retrieved route.
  const secondRoute = scoredRoutes[1] ?? null;
  // Reads the highest route score.
  const topScore = topRoute?.score ?? 0;
  // Reads the second-highest route score.
  const secondScore = secondRoute?.score ?? 0;

  // Returns a structured result object.
  return {
    // Adds this value to the current structure.
    scoringDurationMs,
    // Sets this property in the current object.
    matchedRouteCount: matchedRoutes.length,
    // Sets this property in the current object.
    topRouteId: topRoute?.id ?? null,
    // Adds this value to the current structure.
    topScore,
    // Sets this property in the current object.
    secondRouteId: secondRoute?.id ?? null,
    // Adds this value to the current structure.
    secondScore,
    // Sets this property in the current object.
    scoreMargin: topScore - secondScore,
  };

}

// Builds diagnostic metadata for an assistant response.
function buildRoutingDiagnostics({
  // Adds this value to the current structure.
  startedAt,
  // Adds this value to the current structure.
  routingStrategy,
  // Adds this value to the current structure.
  qwenCalled,
  // Adds this value to the current structure.
  scoreSummary,
  // Assigns the computed value for the current operation.
  qwenCandidateCount = 0,
  // Assigns the computed value for the current operation.
  ollamaMetrics = null,
  // Assigns the computed value for the current operation.
  rankerMetrics = null,
  // Assigns the computed value for the current operation.
  rankerFallbackReason = null,
  // Assigns the computed value for the current operation.
  cacheHit = false,
// Opens the next processing block.
}) {
  // Returns a structured result object.
  return {
    // Adds this value to the current structure.
    routingStrategy,
    // Adds this value to the current structure.
    qwenCalled,
    // Adds this value to the current structure.
    cacheHit,
    // Sets this property in the current object.
    totalDurationMs: elapsedMs(startedAt),
    // Adds this value to the current structure.
    qwenCandidateCount,
    // Adds this value to the current structure.
    scoreSummary,
    // Adds this value to the current structure.
    ollamaMetrics,
    // Adds this value to the current structure.
    rankerMetrics,
    // Adds this value to the current structure.
    rankerFallbackReason,
  };
}

// Writes concise routing and timing diagnostics to the terminal.
function logRoutingDecision(result) {
  // Reads routing diagnostics from the assistant result.
  const diagnostics = result.routingDiagnostics;
  // Checks whether this condition is true.
  if (!diagnostics) return;

  // Reads the route-score summary for logging.
  const summary = diagnostics.scoreSummary ?? {};
  // Writes operational information to the terminal.
  console.log(
    // Continues the current operation.
    [
      // Adds an instruction or value to the current structure.
      "[AMIDS routing]",
      // Assigns the computed value for the current operation.
      `strategy=${diagnostics.routingStrategy}`,
      // Assigns the computed value for the current operation.
      `qwen=${diagnostics.qwenCalled ? "yes" : "no"}`,
      // Assigns the computed value for the current operation.
      `ranker=${diagnostics.rankerMetrics ? "yes" : "no"}`,
      // Assigns the computed value for the current operation.
      `cache=${diagnostics.cacheHit ? "hit" : "miss"}`,
      // Assigns the computed value for the current operation.
      `total=${diagnostics.totalDurationMs}ms`,
      // Assigns the computed value for the current operation.
      `scoring=${summary.scoringDurationMs ?? "n/a"}ms`,
      // Assigns the computed value for the current operation.
      `candidates=${diagnostics.qwenCandidateCount}`,
      // Assigns the computed value for the current operation.
      `top=${summary.topRouteId ?? "none"}`,
      // Assigns the computed value for the current operation.
      `score=${summary.topScore ?? 0}`,
      // Assigns the computed value for the current operation.
      `margin=${summary.scoreMargin ?? 0}`,
    // Continues the current operation.
    ].join(" ")
  );

  // Checks whether this condition is true.
  if (diagnostics.ollamaMetrics) {
    // Reads detailed Ollama timing metrics.
    const metrics = diagnostics.ollamaMetrics;
    // Writes operational information to the terminal.
    console.log(
      // Continues the current operation.
      [
        // Adds an instruction or value to the current structure.
        "[Ollama timing]",
        // Assigns the computed value for the current operation.
        `roundTrip=${metrics.roundTripDurationMs ?? metrics.requestDurationMs}ms`,
        // Assigns the computed value for the current operation.
        `load=${metrics.loadDurationMs ?? "n/a"}ms`,
        // Assigns the computed value for the current operation.
        `prompt=${metrics.promptEvalDurationMs ?? "n/a"}ms`,
        // Assigns the computed value for the current operation.
        `generation=${metrics.evalDurationMs ?? "n/a"}ms`,
        // Assigns the computed value for the current operation.
        `overhead=${metrics.overheadDurationMs ?? "n/a"}ms`,
        // Assigns the computed value for the current operation.
        `promptTokens=${metrics.promptEvalCount ?? "n/a"}`,
        // Assigns the computed value for the current operation.
        `outputTokens=${metrics.evalCount ?? "n/a"}`,
      // Continues the current operation.
      ].join(" ")
    );
  }

  // Writes specialised-ranker confidence and latency when it handled the request.
  if (diagnostics.rankerMetrics) {
    const metrics = diagnostics.rankerMetrics;
    console.log(
      [
        "[PyTorch ranker timing]",
        `roundTrip=${metrics.roundTripDurationMs}ms`,
        `inference=${metrics.inferenceDurationMs ?? "n/a"}ms`,
        `confidence=${metrics.confidence ?? "n/a"}`,
        `scopeProbability=${metrics.scopeProbability ?? "n/a"}`,
        `model=${metrics.modelVersion ?? "unknown"}`,
      ].join(" ")
    );
  }

  // Reports why hybrid mode escalated from PyTorch to Qwen.
  if (diagnostics.rankerFallbackReason) {
    console.log(`[PyTorch ranker fallback] ${diagnostics.rankerFallbackReason}`);
  }
}

// Builds a deterministic fallback response from selected routes.
function buildRouteRegistryResult(payload, selectedRoutes, options) {
  // Removes internal scoring fields from selected routes.
  const publicRoutes = selectedRoutes.map(toPublicRoute);
  // Gets the first selected route as the primary recommendation.
  const firstRoute = selectedRoutes[0] ?? null;

  // Returns a structured result object.
  return {
    // Sets this property in the current object.
    recommendationId: createRecommendationId(
      // Adds this value to the current structure.
      payload,
      // Assigns the computed value for the current operation.
      selectedRoutes.map((route) => route.id)
    // Adds this value to the current structure.
    ),
    // Sets this property in the current object.
    requestQuery: payload.query,
    // Sets this property in the current object.
    mode: options.mode,
    // Sets this property in the current object.
    routingStrategy: options.routingStrategy,
    // Sets this property in the current object.
    roleKey: payload.roleKey,
    // Sets this property in the current object.
    recommendedLinkIds: [],
    // Sets this property in the current object.
    openLinkId: null,
    // Sets this property in the current object.
    route: firstRoute ? toPublicRoute(firstRoute) : null,
    // Sets this property in the current object.
    routes: publicRoutes,
    // Sets this property in the current object.
    openMode: publicRoutes.length > 1 ? "multiple" : "single",
    // Sets this property in the current object.
    shouldOpen: queryRequestsOpen(payload.query) && Boolean(firstRoute),
    // Sets this property in the current object.
    explanation: options.explanation,
    // Sets this property in the current object.
    appliedRules: getAppliedRules(payload),
    // Sets this property in the current object.
    routingDiagnostics: buildRoutingDiagnostics(options),
    // Spreads these values into the current structure.
    ...(options.backendError ? { backendError: options.backendError } : {}),
  };

}

// Scores one route for candidate retrieval and personalization.
function scoreRoute(route, query, roleKey, preferences = {}, expertRules = [], userProfile = {}) {
  // Normalizes the user query for consistent matching.
  const queryText = normalize(query);
  // Splits the normalized query into searchable tokens.
  const queryTokens = tokenize(query);
  // Combines route keywords into searchable text.
  const keywordText = route.keywords.map(normalize).join(" ");
  // Combines route metadata into normalized searchable text.
  const searchableText = normalize([route.title, route.description, keywordText].join(" "));
  // Reads the user's preferred categories.
  const preferredCategories = preferences.preferredCategories ?? [];
  // Reads the user's preferred route IDs.
  const preferredLinks = preferences.preferredLinks ?? [];

  // Scores exact route-keyword phrase matches.
  const exactKeywordScore = route.keywords.reduce(
    // Assigns the computed value for the current operation.
    (score, keyword) => score + (queryText.includes(normalize(keyword)) ? 12 : 0),
    // Continues the current operation.
    0
  );
  // Scores individual query-token matches.
  const tokenScore = queryTokens.reduce((score, token) => {
    // Checks whether this condition is true.
    if (route.title.toLowerCase().includes(token)) return score + 5;
    // Checks whether this condition is true.
    if (keywordText.includes(token)) return score + 4;
    // Checks whether this condition is true.
    if (route.description.toLowerCase().includes(token)) return score + 2;
    // Returns the computed result to the caller.
    return score;
  // Continues the current operation.
  }, 0);
  // Scores matches to saved user preferences.
  const preferenceScore =
    // Assigns the computed value for the current operation.
    preferredCategories.some((category) => searchableText.includes(normalize(category))) ||
    // Continues the current operation.
    preferredLinks.includes(route.id)
      // Continues the current operation.
      ? 3
      // Continues the current operation.
      : 0;
  // Scores routes explicitly associated with the active role.
  const roleScore = route.keywords.includes(roleKey) ? 5 : 0;
  // Applies the bounded learned score for this route.
  const profileRouteScore = Math.max(-6, Math.min(6, userProfile.routeScores?.[route.id] ?? 0));
  // Boosts routes stored as preferred.
  const preferredRouteScore = userProfile.preferredRouteIds?.includes(route.id) ? 6 : 0;
  // Penalizes routes stored as avoided.
  const avoidedRouteScore = userProfile.avoidedRouteIds?.includes(route.id) ? -6 : 0;
  // Scores routes matching the user's frequent topics.
  const frequentTopicScore = (userProfile.frequentTopics ?? []).reduce((score, topic) => {
    // Returns the computed result to the caller.
    return searchableText.includes(normalize(topic)) ? score + 2 : score;
  // Continues the current operation.
  }, 0);
  // Scores routes that match relevant expert-rule instructions.
  const ruleScore = expertRules.reduce((score, rule) => {
    // Checks whether this condition is true.
    if (!rule.trigger) return score;
    // Checks whether the query activates this expert rule.
    const triggerMatches = queryText.includes(normalize(rule.trigger ?? ""));
    // Checks whether the route matches words in the rule instruction.
    const instructionMatches = tokenize(rule.instruction ?? "").some((token) =>
      // Continues the current operation.
      searchableText.includes(token)
    );
    // Returns the computed result to the caller.
    return score + (triggerMatches && instructionMatches ? 8 : 0);
  // Continues the current operation.
  }, 0);

  // Returns the combined expression result.
  return (
    // Continues the current operation.
    exactKeywordScore +
    // Continues the current operation.
    tokenScore +
    // Continues the current operation.
    preferenceScore +
    // Continues the current operation.
    roleScore +
    // Continues the current operation.
    profileRouteScore +
    // Continues the current operation.
    preferredRouteScore +
    // Continues the current operation.
    avoidedRouteScore +
    // Continues the current operation.
    frequentTopicScore +
    // Continues the current operation.
    ruleScore
  );
}

// Scores and sorts all approved registry routes.
function scoreCandidateRoutes(payload) {
  // Returns routes from the approved registry.
  return routeRegistry
    // Transforms each entry into the required shape.
    .map((route) => ({
      // Spreads these values into the current structure.
      ...route,
      // Sets this property in the current object.
      relevanceScore: scoreRoute(route, payload.query, payload.roleKey),
      // Sets this property in the current object.
      score: scoreRoute(
        // Adds this value to the current structure.
        route,
        // Adds this value to the current structure.
        payload.query,
        // Adds this value to the current structure.
        payload.roleKey,
        // Adds this value to the current structure.
        payload.preferences ?? {},
        // Adds this value to the current structure.
        payload.expertRules ?? [],
        // Continues the current operation.
        payload.userProfile ?? {}
      // Adds this value to the current structure.
      ),
    // Continues the current operation.
    }))
    // Sorts entries into the required order.
    .sort((a, b) => b.score - a.score);
}

// Keeps candidate routes with positive scores.
function getMatchedRoutes(scoredRoutes) {
  // Returns the computed result to the caller.
  return scoredRoutes.filter((route) => route.score > 0);
}

// Builds a high-recall shortlist for the AI model.
function findCandidateRoutes(scoredRoutes, candidateLimit = qwenCandidateLimit) {
  // Sorts candidates by query relevance without personalization.
  const relevanceRoutes = [...scoredRoutes].sort(
    // Assigns the computed value for the current operation.
    (routeA, routeB) => routeB.relevanceScore - routeA.relevanceScore
  );
  // Reserves most candidate slots for query-relevant routes.
  const relevanceQuota = Math.ceil(candidateLimit * 0.75);
  // Stores unique candidate routes by ID.
  const candidateRoutesById = new Map();

  // Processes each entry in the current list.
  [...relevanceRoutes.slice(0, relevanceQuota), ...scoredRoutes].forEach((route) => {
    // Checks whether this condition is true.
    if (candidateRoutesById.size < candidateLimit && !candidateRoutesById.has(route.id)) {
      // Stores this value in the current map or cache.
      candidateRoutesById.set(route.id, route);
    }
  });
  // Returns a newly constructed list.
  return [...candidateRoutesById.values()];
}

// Keeps only compact expert rules relevant to this query.
function compactRelevantExpertRules(payload) {
  // Normalizes the user query for consistent matching.
  const queryText = normalize(payload.query);

  // Returns the combined expression result.
  return (payload.expertRules ?? [])
    // Keeps only entries that satisfy this condition.
    .filter((rule) => rule.trigger && queryText.includes(normalize(rule.trigger)))
    // Limits the current value to the required size.
    .slice(0, 3)
    // Transforms each entry into the required shape.
    .map((rule) => ({
      // Sets this property in the current object.
      name: rule.name,
      // Sets this property in the current object.
      trigger: rule.trigger,
      // Sets this property in the current object.
      instruction: String(rule.instruction ?? "").slice(0, 160),
    // Continues the current operation.
    }));
}

// Keeps only compact personalization hints for the model.
function compactPersonalization(payload) {
  // Returns a structured result object.
  return {
    // Sets this property in the current object.
    preferredCategories: (payload.preferences?.preferredCategories ?? []).slice(0, 4),
    // Sets this property in the current object.
    preferredRouteIds: (payload.userProfile?.preferredRouteIds ?? []).slice(0, 6),
    // Sets this property in the current object.
    avoidedRouteIds: (payload.userProfile?.avoidedRouteIds ?? []).slice(0, 4),
    // Sets this property in the current object.
    frequentTopics: (payload.userProfile?.frequentTopics ?? []).slice(0, 6),
  };
}

// Removes internal route fields before sending a candidate to Ollama.
function compactCandidateRoute(route, query) {
  // Normalizes the user query for consistent matching.
  const queryText = normalize(query);
  // Splits the normalized query into searchable tokens.
  const queryTokens = tokenize(query);
  // Ranks route keywords by relevance to the current query.
  const rankedKeywords = unique(route.keywords)
    // Transforms each entry into the required shape.
    .map((keyword, index) => {
      // Normalizes the current route keyword.
      const normalizedKeyword = normalize(keyword);
      // Scores whether the full keyword appears in the query.
      const exactPhraseMatch = queryText.includes(normalizedKeyword) ? 20 : 0;
      // Counts query tokens contained in this keyword.
      const tokenMatches = queryTokens.filter((token) => normalizedKeyword.includes(token)).length;
      // Returns a structured result object.
      return {
        // Adds this value to the current structure.
        keyword,
        // Sets this property in the current object.
        score: exactPhraseMatch + tokenMatches * 4 - index / 100,
      };
    })
    // Sorts entries into the required order.
    .sort((a, b) => b.score - a.score)
    // Limits the current value to the required size.
    .slice(0, 6)
    // Transforms each entry into the required shape.
    .map(({ keyword }) => keyword);

  // Returns a structured result object.
  return {
    // Sets this property in the current object.
    id: route.id,
    // Sets this property in the current object.
    title: route.title,
    // Sets this property in the current object.
    purpose: String(route.description ?? "").slice(0, 160),
    // Sets this property in the current object.
    keywords: rankedKeywords,
  };
}

// Builds a hash key for an assistant request.
function buildAssistantCacheKey(payload) {
  // Returns a deterministic SHA-256 hash.
  return createHash("sha256")
    // Adds normalized data to the hash input.
    .update(
      // Opens the next processing block.
      JSON.stringify({
        // Sets this property in the current object.
        query: normalize(payload.query),
        // Sets this property in the current object.
        roleKey: payload.roleKey ?? null,
        // Sets this property in the current object.
        relevantRules: compactRelevantExpertRules(payload),
        // Sets this property in the current object.
        personalization: compactPersonalization(payload),
        // Sets this property in the current object.
        profileUpdatedAt: payload.userProfile?.updatedAt ?? null,
        // Sets this property in the current object.
        model: OLLAMA_ROUTING_MODEL,
        // Adds this value to the current structure.
        routingProvider,
        // Sets this property in the current object.
        promptVersion: assistantPromptVersion,
        // Adds this value to the current structure.
        routeRegistryVersion,
      })
    // Continues the current operation.
    )
    // Converts the completed hash to hexadecimal text.
    .digest("hex");
}

// Returns a valid cached assistant response when available.
function getCachedAssistantResult(cacheKey) {
  // Reads a previously cached assistant result.
  const cached = assistantResponseCache.get(cacheKey);
  // Checks whether this condition is true.
  if (!cached) return null;

  // Checks whether this condition is true.
  if (cached.expiresAt <= Date.now()) {
    // Removes this entry from the current map or cache.
    assistantResponseCache.delete(cacheKey);
    // Returns the computed result to the caller.
    return null;
  }

  // Removes this entry from the current map or cache.
  assistantResponseCache.delete(cacheKey);
  // Stores this value in the current map or cache.
  assistantResponseCache.set(cacheKey, cached);
  // Returns the computed result to the caller.
  return cached.result;
}

// Stores a validated assistant response in the bounded cache.
function cacheAssistantResult(cacheKey, result) {
  // Checks whether this condition is true.
  if (assistantCacheTtlMs <= 0) return;

  // Stores this value in the current map or cache.
  assistantResponseCache.set(cacheKey, {
    // Sets this property in the current object.
    expiresAt: Date.now() + assistantCacheTtlMs,
    // Adds this value to the current structure.
    result,
  });

  // Repeats while the cache exceeds its configured limit.
  while (assistantResponseCache.size > assistantCacheMaxEntries) {
    // Gets the oldest cache entry for eviction.
    const oldestCacheKey = assistantResponseCache.keys().next().value;
    // Removes this entry from the current map or cache.
    assistantResponseCache.delete(oldestCacheKey);
  }
}

// Returns a cached or shared result with updated diagnostics.
function reuseAssistantResult(result, startedAt, routingStrategy, cacheHit) {
  // Returns a structured result object.
  return {
    // Spreads these values into the current structure.
    ...result,
    // Adds this value to the current structure.
    routingStrategy,
    // Sets this property in the current object.
    routingDiagnostics: {
      // Spreads these values into the current structure.
      ...(result.routingDiagnostics ?? {}),
      // Adds this value to the current structure.
      routingStrategy,
      // Sets this property in the current object.
      qwenCalled: false,
      // Adds this value to the current structure.
      cacheHit,
      // Sets this property in the current object.
      totalDurationMs: elapsedMs(startedAt),
      // Sets this property in the current object.
      ollamaMetrics: null,
      // Sets this property in the current object.
      rankerMetrics: null,
      // Sets this property in the current object.
      rankerFallbackReason: null,
    },
  };
}

// Parses JSON from the model response, including JSON embedded in prose.
function extractJson(content) {
  // Removes surrounding whitespace from model output.
  const trimmed = content.trim();

  // Starts an operation that may fail.
  try {
    // Returns parsed JSON data.
    return JSON.parse(trimmed);
  // Handles an error from the preceding operation.
  } catch {
    // Extracts a JSON-looking object from model output.
    const match = trimmed.match(/\{[\s\S]*\}/);
    // Checks whether this condition is true.
    if (!match) {
      // Stops processing with a descriptive error.
      throw new Error("Ollama did not return JSON.");
    }
    // Returns parsed JSON data.
    return JSON.parse(match[0]);
  }
}

// Validates model-selected route IDs and builds a safe frontend response.
function sanitizeResult(rawResult, payload, candidateRoutes, metadata) {
  // Preserves explicit all/every wording as a user constraint even if the model misclassifies it.
  const explicitMultipleRoutes = queryRequestsAllRelated(payload.query);
  // Uses either the explicit user constraint or the model's inferred scope.
  const wantsMultipleRoutes = explicitMultipleRoutes || rawResult.requestScope === "multiple";
  // Creates a set of route IDs the AI is permitted to select.
  const candidateRouteIds = new Set(candidateRoutes.map((route) => route.id));
  // Creates a lookup from approved route IDs to route objects.
  const routesById = new Map(candidateRoutes.map((route) => [route.id, route]));
  // Collects the route IDs requested by the model.
  const requestedRouteIds = unique([
    // Spreads these values into the current structure.
    ...(wantsMultipleRoutes && Array.isArray(rawResult.routeIds) ? rawResult.routeIds : []),
    // Adds this value to the current structure.
    rawResult.routeId,
    // Spreads these values into the current structure.
    ...(!wantsMultipleRoutes && Array.isArray(rawResult.routeIds) ? rawResult.routeIds : []),
  // Continues the current operation.
  ]);
  // Checks whether the model's primary route ID is approved.
  const hasValidSingleRouteId =
    // Assigns the computed value for the current operation.
    typeof rawResult.routeId === "string" && candidateRouteIds.has(rawResult.routeId);
  // Tracks route IDs already accepted from the model.
  const seenRouteIds = new Set();
  // Validates and resolves the route IDs selected by the model.
  const selectedRoutesFromAi = requestedRouteIds
    // Keeps only entries that satisfy this condition.
    .filter((routeId) => {
      // Checks whether this condition is true.
      if (!candidateRouteIds.has(routeId) || seenRouteIds.has(routeId)) return false;
      // Continues the current operation.
      seenRouteIds.add(routeId);
      // Returns the computed result to the caller.
      return true;
    })
    // Transforms each entry into the required shape.
    .map((routeId) => routesById.get(routeId))
    // Keeps only entries that satisfy this condition.
    .filter(Boolean)
    // Limits the current value to the required size.
    .slice(0, wantsMultipleRoutes ? maxRelatedRoutes : 1);
  // Requires at least two approved selections when the user explicitly requests multiple routes.
  const minimumSelectedRoutes = wantsMultipleRoutes ? Math.min(2, candidateRoutes.length) : 1;
  // Records whether the model selected enough approved routes for the requested scope.
  const hasValidAiSelection = wantsMultipleRoutes
    ? selectedRoutesFromAi.length >= minimumSelectedRoutes
    : hasValidSingleRouteId || selectedRoutesFromAi.length > 0;
  // Chooses the approved primary route or strongest fallback candidate.
  const selectedRoute =
    // Continues the current operation.
    hasValidSingleRouteId
      // Continues the current operation.
      ? routesById.get(rawResult.routeId)
      // Continues the current operation.
      : selectedRoutesFromAi[0] ?? candidateRoutes[0] ?? null;
  // Builds the final single-route or multiple-route selection.
  const selectedRoutes = hasValidAiSelection
    ? wantsMultipleRoutes
      ? selectedRoutesFromAi
      : selectedRoute
        ? [selectedRoute]
        : []
    : wantsMultipleRoutes
      ? candidateRoutes.slice(0, maxRelatedRoutes)
      : selectedRoute
        ? [selectedRoute]
        : [];
  // Removes internal scoring fields from selected routes.
  const publicRoutes = selectedRoutes.map(toPublicRoute);
  // Gets the first selected route as the primary recommendation.
  const firstRoute = selectedRoutes[0] ?? selectedRoute ?? null;
  // Builds an explanation for an invalid model selection.
  const fallbackExplanation = firstRoute
    // Continues the current operation.
    ? "The AI ranker did not select a valid approved route ID, so the strongest retrieved routeRegistry candidate was used."
    // Continues the current operation.
    : "The AI ranker did not select a valid approved route ID and no routeRegistry candidate was available.";
  // Uses the model explanation or a safe generated explanation.
  const explanation = hasValidAiSelection
    // Continues the current operation.
    ? rawResult.explanation ??
      // Continues the current operation.
      (publicRoutes.length > 1
        // Continues the current operation.
        ? `The AI ranker matched ${publicRoutes.length} approved routeRegistry routes.`
        // Continues the current operation.
        : "The AI ranker matched the request to an approved routeRegistry route.")
    // Continues the current operation.
    : fallbackExplanation;
  // Labels whether the model selection or a fallback was used.
  const routingStrategy = hasValidAiSelection
    // Continues the current operation.
    ? metadata.routingStrategy
    // Continues the current operation.
    : "deterministic-invalid-ai-fallback";

  // Returns a structured result object.
  return {
    // Sets this property in the current object.
    recommendationId: createRecommendationId(
      // Adds this value to the current structure.
      payload,
      // Assigns the computed value for the current operation.
      selectedRoutes.map((route) => route.id)
    // Adds this value to the current structure.
    ),
    // Sets this property in the current object.
    requestQuery: payload.query,
    // Sets this property in the current object.
    mode: metadata.mode ?? "ollama-qwen3-local",
    // Adds this value to the current structure.
    routingStrategy,
    // Sets this property in the current object.
    roleKey: payload.roleKey,
    // Sets this property in the current object.
    recommendedLinkIds: [],
    // Sets this property in the current object.
    openLinkId: null,
    // Sets this property in the current object.
    route: firstRoute ? toPublicRoute(firstRoute) : null,
    // Sets this property in the current object.
    routes: publicRoutes,
    // Sets this property in the current object.
    openMode: wantsMultipleRoutes && publicRoutes.length > 1 ? "multiple" : "single",
    // Sets this property in the current object.
    shouldOpen: Boolean((rawResult.shouldOpen || queryRequestsOpen(payload.query)) && firstRoute),
    // Adds this value to the current structure.
    explanation,
    // Sets this property in the current object.
    appliedRules:
      // Adds this value to the current structure.
      hasValidAiSelection && Array.isArray(rawResult.appliedRules) ? rawResult.appliedRules : [],
    // Sets this property in the current object.
    routingDiagnostics: buildRoutingDiagnostics({
      // Spreads these values into the current structure.
      ...metadata,
      // Adds this value to the current structure.
      routingStrategy,
    // Adds this value to the current structure.
    }),
  };

}

// Converts stored preferences and expert rules into small route-logit adjustments.
function buildRankerRouteBiases(payload) {
  // Stores only non-zero route adjustments.
  const biases = {};
  // Normalizes the current query for expert-rule matching.
  const queryText = normalize(payload.query);

  // Calculates one bounded adjustment for every approved route.
  routeRegistry.forEach((route) => {
    // Reads the bounded learned personalization score.
    const profileScore = Math.max(-6, Math.min(6, payload.userProfile?.routeScores?.[route.id] ?? 0));
    // Starts with a small learned-profile adjustment.
    let bias = profileScore / 6;

    // Applies explicit preference lists.
    if (payload.userProfile?.preferredRouteIds?.includes(route.id)) bias += 0.5;
    if (payload.userProfile?.avoidedRouteIds?.includes(route.id)) bias -= 0.5;
    if (payload.preferences?.preferredLinks?.includes(route.id)) bias += 0.5;

    // Applies only rules whose trigger matches the current query.
    (payload.expertRules ?? []).forEach((rule) => {
      if (!rule.trigger || !queryText.includes(normalize(rule.trigger))) return;
      if (rule.linkIds?.includes(route.id)) bias += 1;
    });

    // Sends only meaningful bounded adjustments to the ranker.
    if (bias !== 0) biases[route.id] = Math.max(-2, Math.min(2, bias));
  });

  // Returns compact route-specific adjustments.
  return biases;
}

// Asks the specialised local PyTorch service to rank every approved route.
async function askPytorchRanker(payload, metadata) {
  // Records the complete local ranker round-trip duration.
  const rankerStartedAt = Date.now();
  // Creates a timeout so hybrid mode can fall back promptly.
  const controller = new AbortController();
  // Schedules cancellation when the ranker exceeds its configured limit.
  const timeout = setTimeout(() => controller.abort(), pytorchRankerTimeoutMs);

  // Starts an operation that may fail.
  try {
    // Sends only query context and bounded personalization adjustments.
    const response = await fetch(PYTORCH_RANKER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        query: payload.query,
        roleKey: payload.roleKey ?? null,
        maxRoutes: maxRelatedRoutes,
        routeBiases: buildRankerRouteBiases(payload),
      }),
    });

    // Rejects an unsuccessful ranker HTTP response.
    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(
        `PyTorch ranker request failed with status ${response.status}: ${
          errorPayload.detail ?? errorPayload.error ?? "unknown error"
        }`
      );
    }

    // Parses the ranker's structured decision.
    const rawResult = await response.json();
    // Escalates uncertain decisions instead of silently opening a weak match.
    if (rawResult.needsFallback) {
      throw new Error(
        `PyTorch ranker confidence was insufficient (${rawResult.confidence ?? "unknown"}).`
      );
    }

    // Validates ranker-selected IDs against Node's authoritative registry.
    return sanitizeResult(
      {
        ...rawResult,
        appliedRules: getAppliedRules(payload),
      },
      payload,
      routeRegistry,
      {
        ...metadata,
        mode: "pytorch-route-ranker-local",
        routingStrategy: "ai-pytorch-route-ranking",
        qwenCalled: false,
        qwenCandidateCount: routeRegistry.length,
        rankerMetrics: {
          roundTripDurationMs: elapsedMs(rankerStartedAt),
          inferenceDurationMs: rawResult.durationMs ?? null,
          confidence: rawResult.confidence ?? null,
          scopeProbability: rawResult.scopeProbability ?? null,
          modelVersion: rawResult.modelVersion ?? null,
        },
      }
    );
  // Runs cleanup whether the request succeeds or fails.
  } finally {
    // Cancels the timeout once the ranker request finishes.
    clearTimeout(timeout);
  }
}

// Asks the configured Ollama model to select approved routes.
async function askOllama(payload, candidateRoutes, metadata) {
  // Detects an explicit user instruction to return all or every related route.
  const explicitMultipleRoutes = queryRequestsAllRelated(payload.query);
  // Defines the structured JSON response Ollama must return.
  const schema = {
    // Sets this property in the current object.
    type: "object",
    // Sets this property in the current object.
    additionalProperties: false,
    // Sets this property in the current object.
    properties: {
      // Sets this property in the current object.
      shouldOpen: { type: "boolean" },
      // Sets this property in the current object.
      requestScope: {
        // Sets this property in the current object.
        type: "string",
        // Forces multiple scope when the user explicitly asks for all/every data.
        enum: explicitMultipleRoutes ? ["multiple"] : ["single", "multiple"],
      },
      // Sets this property in the current object.
      explanation: { type: "string" },
      // Sets this property in the current object.
      appliedRules: {
        // Sets this property in the current object.
        type: "array",
        // Sets this property in the current object.
        items: { type: "string" },
      },
      // Sets this property in the current object.
      routeId: { type: ["string", "null"] },
      // Sets this property in the current object.
      routeIds: {
        // Sets this property in the current object.
        type: "array",
        // Sets this property in the current object.
        items: { type: "string" },
        // Requires more than one route for an explicit all/every request.
        minItems: explicitMultipleRoutes ? Math.min(2, candidateRoutes.length) : 0,
        // Sets this property in the current object.
        maxItems: maxRelatedRoutes,
      },
    },
    // Sets this property in the current object.
    required: [
      // Adds an instruction or value to the current structure.
      "shouldOpen",
      // Adds an instruction or value to the current structure.
      "requestScope",
      // Adds an instruction or value to the current structure.
      "explanation",
      // Adds an instruction or value to the current structure.
      "appliedRules",
      // Adds an instruction or value to the current structure.
      "routeId",
      // Adds an instruction or value to the current structure.
      "routeIds",
    // Adds this value to the current structure.
    ],
  };

  // Records when the Ollama request begins.
  const qwenStartedAt = Date.now();
  // Sends the structured route-selection request to Ollama.
  const response = await fetch(OLLAMA_URL, {
    // Sets this property in the current object.
    method: "POST",
    // Sets this property in the current object.
    headers: {
      // Adds an instruction or value to the current structure.
      "Content-Type": "application/json",
    },
    // Sets this property in the current object.
    body: JSON.stringify({
      // Sets this property in the current object.
      model: OLLAMA_ROUTING_MODEL,
      // Sets this property in the current object.
      stream: false,
      // Sets this property in the current object.
      think: false,
      // Adds this value to the current structure.
      keep_alive: ollamaKeepAlive,
      // Sets this property in the current object.
      format: schema,
      // Sets this property in the current object.
      options: {
        // Sets this property in the current object.
        temperature: 0,
        // Adds this value to the current structure.
        num_ctx: explicitMultipleRoutes ? broadOllamaNumCtx : ollamaNumCtx,
        // Adds this value to the current structure.
        num_predict: ollamaNumPredict,
      },
      // Sets this property in the current object.
      messages: [
        // Opens the next processing block.
        {
          // Sets this property in the current object.
          role: "system",
          // Sets this property in the current object.
          content: [
            // Adds an instruction or value to the current structure.
            "You are the primary local AMIDS navigation-intent and route-selection assistant.",
            // Adds an instruction or value to the current structure.
            "Choose route IDs only from candidateRoutes.",
            // Adds an instruction or value to the current structure.
            "Interpret the user's meaning, qualifiers, exclusions, and whether they want one source or multiple related sources.",
            // Adds an instruction or value to the current structure.
            "When requestScopeConstraint is multiple, it is an explicit user instruction: requestScope must be multiple and routeIds must contain multiple directly relevant routes.",
            // Adds an instruction or value to the current structure.
            "For a single-source request, set requestScope to single, put the best ID in routeId, and return routeIds as an empty array.",
            // Adds an instruction or value to the current structure.
            `For a request for all, every, complete, comprehensive, or otherwise multiple related sources, set requestScope to multiple and return up to ${maxRelatedRoutes} coherent IDs in routeIds.`,
            // Adds an instruction or value to the current structure.
            "For multiple-source requests, include routes that directly cover the requested subject and exclude merely tangential dashboards or alerts unless the user explicitly asks for them.",
            // Adds an instruction or value to the current structure.
            "Use purpose and keywords to distinguish routes with similar titles.",
            // Adds an instruction or value to the current structure.
            "Treat relevant expert rules as explicit instructions and personalization as a preference, while prioritizing the user's current request.",
            // Adds an instruction or value to the current structure.
            "Always choose the closest candidate route when no candidate is a perfect match.",
            // Adds an instruction or value to the current structure.
            "Do not answer that none of the routes are relevant.",
            // Adds an instruction or value to the current structure.
            "Never invent route IDs, URLs, weather values, or operational decisions.",
            // Adds an instruction or value to the current structure.
            "If the user asks to open, go to, navigate to, take them to, show, display, view, load, get, give, need, or find an item, set shouldOpen to true.",
            // Adds an instruction or value to the current structure.
            "If the user only asks for information, choose the best route but set shouldOpen to false.",
            // Adds an instruction or value to the current structure.
            "Keep the explanation under 20 words.",
            // Adds an instruction or value to the current structure.
            "Return JSON only and follow the schema exactly.",
          // Adds this value to the current structure.
          ].join(" "),
        },
        // Opens the next processing block.
        {
          // Sets this property in the current object.
          role: "user",
          // Sets this property in the current object.
          content: JSON.stringify({
            // Sets this property in the current object.
            query: payload.query,
            // Sets this property in the current object.
            roleKey: payload.roleKey,
            // Sets this property in the current object.
            requestScopeConstraint: explicitMultipleRoutes ? "multiple" : "model-decides",
            // Sets this property in the current object.
            expertRules: compactRelevantExpertRules(payload),
            // Sets this property in the current object.
            personalization: compactPersonalization(payload),
            // Sets this property in the current object.
            candidateRoutes: candidateRoutes.map((route) =>
              // Continues the current operation.
              compactCandidateRoute(route, payload.query)
            // Adds this value to the current structure.
            ),
            // Sets this property in the current object.
            allowedRouteIds: candidateRoutes.map((route) => route.id),
          // Adds this value to the current structure.
          }),
        },
      // Adds this value to the current structure.
      ],
    // Adds this value to the current structure.
    }),
  });

  // Rejects an unsuccessful Ollama HTTP response.
  if (!response.ok) {
    // Stops processing with a descriptive error.
    throw new Error(`Ollama request failed with status ${response.status}`);
  }

  // Parses the JSON response returned by Ollama.
  const data = await response.json();
  // Extracts the model's structured route-selection result.
  const rawResult = extractJson(data.message?.content ?? "");
  // Returns the computed result to the caller.
  return sanitizeResult(rawResult, payload, candidateRoutes, {
    // Spreads these values into the current structure.
    ...metadata,
    // Sets this property in the current object.
    qwenCandidateCount: candidateRoutes.length,
    // Sets this property in the current object.
    ollamaMetrics: summarizeOllamaMetrics(data, qwenStartedAt),
  });

}

// Retrieves candidates and performs one uncached model decision.
async function answerAssistantRequestUncached(payload, startedAt) {
  // Detects whether this request explicitly requires a multiple-route answer.
  const explicitMultipleRoutes = queryRequestsAllRelated(payload.query);
  // Records when candidate retrieval begins.
  const scoringStartedAt = Date.now();
  // Scores every approved route for shortlist retrieval.
  const scoredRoutes = scoreCandidateRoutes(payload);
  // Keeps routes with a positive retrieval score.
  const matchedRoutes = getMatchedRoutes(scoredRoutes);
  // Summarizes shortlist scoring for diagnostics.
  const scoreSummary = buildScoreSummary(
    // Adds this value to the current structure.
    scoredRoutes,
    // Adds this value to the current structure.
    matchedRoutes,
    // Continues the current operation.
    elapsedMs(scoringStartedAt)
  );

  // Stores a PyTorch failure so hybrid mode can report it if Qwen also fails.
  let rankerError = null;
  // Uses the specialised model first unless Ollama-only mode is configured.
  if (routingProvider !== "ollama") {
    // Starts an operation that may fail.
    try {
      // Returns a confident specialised-model decision immediately.
      return await askPytorchRanker(payload, {
        startedAt,
        scoreSummary,
      });
    // Handles an unavailable or uncertain specialised-model decision.
    } catch (error) {
      // Stores the failure for diagnostics and possible fallback.
      rankerError = error;

      // Avoids calling Qwen when strict PyTorch-only mode is configured.
      if (routingProvider === "pytorch") {
        const fallbackSourceRoutes = matchedRoutes.length > 0 ? matchedRoutes : scoredRoutes;
        const fallbackRoutes = explicitMultipleRoutes
          ? fallbackSourceRoutes.slice(0, maxRelatedRoutes)
          : fallbackSourceRoutes.slice(0, 1);

        // Returns an emergency registry result when the specialised service cannot decide.
        return buildRouteRegistryResult(payload, fallbackRoutes, {
          mode: "deterministic-route-registry",
          routingStrategy: "deterministic-pytorch-fallback",
          qwenCalled: false,
          qwenCandidateCount: routeRegistry.length,
          startedAt,
          scoreSummary,
          backendError: `PyTorch routing failed: ${error.message}`,
          explanation: fallbackRoutes.length > 0
            ? "The specialised PyTorch ranker was unavailable or uncertain, so emergency routeRegistry scoring was used."
            : "The specialised PyTorch ranker was unavailable or uncertain and no fallback route was found.",
        });
      }
    }
  }

  // Builds the compact shortlist passed to the AI model.
  const candidateRoutes = findCandidateRoutes(
    scoredRoutes,
    explicitMultipleRoutes ? broadQwenCandidateLimit : qwenCandidateLimit
  );
  // Starts an operation that may fail.
  try {
    // Waits for and returns the asynchronous result.
    return await askOllama(payload, candidateRoutes, {
      // Sets this property in the current object.
      mode: "ollama-qwen3-local",
      // Sets this property in the current object.
      routingStrategy: "ai-qwen-route-selection",
      // Sets this property in the current object.
      qwenCalled: true,
      // Adds this value to the current structure.
      startedAt,
      // Adds this value to the current structure.
      scoreSummary,
      // Sets this property in the current object.
      rankerFallbackReason: rankerError?.message ?? null,
    });
  // Handles an error from the preceding operation.
  } catch (error) {
    // Returns several retrieved routes for broad requests and one route for specific requests.
    const fallbackSourceRoutes = matchedRoutes.length > 0 ? matchedRoutes : candidateRoutes;
    // Selects the fallback route count that matches the user's requested scope.
    const fallbackRoutes = explicitMultipleRoutes
      ? fallbackSourceRoutes.slice(0, maxRelatedRoutes)
      : fallbackSourceRoutes.slice(0, 1);
    // Returns the computed result to the caller.
    return buildRouteRegistryResult(payload, fallbackRoutes, {
      // Sets this property in the current object.
      mode: "deterministic-route-registry",
      // Sets this property in the current object.
      routingStrategy: "deterministic-qwen-fallback",
      // Sets this property in the current object.
      qwenCalled: true,
      // Sets this property in the current object.
      qwenCandidateCount: candidateRoutes.length,
      // Adds this value to the current structure.
      startedAt,
      // Adds this value to the current structure.
      scoreSummary,
      // Sets this property in the current object.
      backendError: [
        rankerError ? `PyTorch ranker did not handle the request: ${rankerError.message}` : null,
        `Qwen routing failed, so deterministic routeRegistry scoring was used: ${error.message}`,
      ].filter(Boolean).join(" "),
      // Sets this property in the current object.
      explanation: fallbackRoutes.length > 0
        ? explicitMultipleRoutes
          ? `Qwen did not return a valid multiple-route selection, so ${fallbackRoutes.length} retrieved routeRegistry matches were used as an emergency fallback.`
          : "Qwen did not return a valid route selection, so the strongest retrieved routeRegistry match was used as an emergency fallback."
        : "Qwen did not return a valid route selection and no routeRegistry fallback was available.",
    });
  }
}

// Serves cached requests or coordinates one new model decision.
async function answerAssistantRequest(payload) {
  // Records when assistant request processing begins.
  const startedAt = Date.now();
  // Creates a cache identity for this request and personalization state.
  const cacheKey = buildAssistantCacheKey(payload);
  // Looks for an existing validated response in the cache.
  const cachedResult = getCachedAssistantResult(cacheKey);

  // Returns the validated cached result without calling the model again.
  if (cachedResult) {
    // Returns the computed result to the caller.
    return reuseAssistantResult(cachedResult, startedAt, "ai-response-cache", true);
  }

  // Looks for an identical model request already in progress.
  const inFlightRequest = assistantInFlightRequests.get(cacheKey);
  // Reuses an identical request that is already waiting for the model.
  if (inFlightRequest) {
    // Stores the completed assistant or feedback result.
    const result = await inFlightRequest;
    // Returns the computed result to the caller.
    return reuseAssistantResult(result, startedAt, "ai-in-flight-reuse", false);
  }

  // Starts one uncached AI routing request.
  const requestPromise = answerAssistantRequestUncached(payload, startedAt);
  // Stores this value in the current map or cache.
  assistantInFlightRequests.set(cacheKey, requestPromise);

  // Starts an operation that may fail.
  try {
    // Stores the completed assistant or feedback result.
    const result = await requestPromise;
    // Caches only successful AI-selected responses.
    if (
      ["ai-qwen-route-selection", "ai-pytorch-route-ranking"].includes(result.routingStrategy)
    ) {
      // Continues the current operation.
      cacheAssistantResult(cacheKey, result);
    }
    // Returns the computed result to the caller.
    return result;
  // Runs cleanup whether the operation succeeds or fails.
  } finally {
    // Removes this entry from the current map or cache.
    assistantInFlightRequests.delete(cacheKey);
  }
}

// Creates the local HTTP API server.
const server = http.createServer(async (request, response) => {
  // Handles the browser's CORS preflight request.
  if (request.method === "OPTIONS") {
    // Writes the HTTP status and response headers.
    response.writeHead(204, jsonHeaders);
    // Finishes the HTTP response.
    response.end();
    // Continues the current operation.
    return;
  }

  // Parses the incoming request URL.
  const requestUrl = new URL(request.url ?? "/", `http://${request.headers.host ?? "127.0.0.1"}`);

  // Handles requests for API health and active routing configuration.
  if (request.method === "GET" && requestUrl.pathname === "/api/health") {
    // Sends the API response as JSON.
    sendJson(response, 200, {
      // Sets this property in the current object.
      ok: true,
      // Sets this property in the current object.
      model: OLLAMA_ROUTING_MODEL,
      // Sets this property in the current object.
      ollamaUrl: OLLAMA_URL,
      // Sets this property in the current object.
      pytorchRankerUrl: PYTORCH_RANKER_URL,
      // Sets this property in the current object.
      registeredRoutes: routeRegistry.length,
      // Sets this property in the current object.
      routing: {
        // Sets this property in the current object.
        strategy: routingProvider,
        // Adds this value to the current structure.
        routingProvider,
        // Adds this value to the current structure.
        pytorchRankerTimeoutMs,
        // Adds this value to the current structure.
        qwenCandidateLimit,
        // Adds this value to the current structure.
        broadQwenCandidateLimit,
        // Adds this value to the current structure.
        maxRelatedRoutes,
        // Adds this value to the current structure.
        assistantCacheTtlMs,
        // Adds this value to the current structure.
        assistantCacheMaxEntries,
        // Sets this property in the current object.
        assistantCacheEntries: assistantResponseCache.size,
        // Sets this property in the current object.
        inFlightRequests: assistantInFlightRequests.size,
        // Adds this value to the current structure.
        ollamaKeepAlive,
        // Adds this value to the current structure.
        ollamaNumCtx,
        // Adds this value to the current structure.
        broadOllamaNumCtx,
        // Adds this value to the current structure.
        ollamaNumPredict,
      },
    });
    // Continues the current operation.
    return;
  }

  // Handles requests to load one user-role personalization profile.
  if (request.method === "GET" && requestUrl.pathname === "/api/amids-assistant/profile") {
    // Uses the supplied user ID or the prototype fallback ID.
    const userId = requestUrl.searchParams.get("userId") || defaultUserId;
    // Uses the supplied role key or an unknown-role fallback.
    const roleKey = requestUrl.searchParams.get("roleKey") || "unknown-role";
    // Gets or creates the personalization profile for this user and role.
    const profile = await getPersonalizationProfile(userId, roleKey);
    // Sends the API response as JSON.
    sendJson(response, 200, {
      // Sets this property in the current object.
      ok: true,
      // Sets this property in the current object.
      mode: "profile-read-local",
      // Adds this value to the current structure.
      profile,
    });
    // Continues the current operation.
    return;
  }

  // Handles one helpful or not-helpful feedback submission.
  if (request.method === "POST" && requestUrl.pathname === "/api/amids-assistant/feedback") {
    // Starts an operation that may fail.
    try {
      // Parses the incoming JSON request payload.
      const payload = await readBody(request);
      // Stores the completed assistant or feedback result.
      const result = await processFeedback(payload);
      // Sends the API response as JSON.
      sendJson(response, 200, result);
    // Handles an error from the preceding operation.
    } catch (error) {
      // Sends the API response as JSON.
      sendJson(response, 500, {
        // Sets this property in the current object.
        ok: false,
        // Sets this property in the current object.
        error: error.message,
        // Sets this property in the current object.
        mode: "profile-update-error",
      });
    }
    // Continues the current operation.
    return;

  }

  // Handles simulated routeRegistry destination pages.
  if (request.method === "GET" && requestUrl.pathname.startsWith("/amids/routes/")) {
    // Extracts the requested route ID from the URL.
    const routeId = decodeURIComponent(requestUrl.pathname.replace("/amids/routes/", ""));
    // Finds the matching approved routeRegistry entry.
    const route = routeRegistry.find((item) => item.id === routeId);

    // Returns a not-found page when the route ID is not approved.
    if (!route) {
      // Sends the simulated route page as HTML.
      sendHtml(
        // Adds this value to the current structure.
        response,
        // Adds this value to the current structure.
        404,
        // Adds an instruction or value to the current structure.
        "<!doctype html><h1>Route not found</h1><p>The requested AMIDS registry route does not exist.</p>"
      );
      // Continues the current operation.
      return;
    }

    // Sends the simulated route page as HTML.
    sendHtml(response, 200, renderRoutePage(route));
    // Continues the current operation.
    return;
  }

  // Handles the main assistant route-selection request.
  if (request.method === "POST" && requestUrl.pathname === "/api/amids-assistant") {
    // Starts an operation that may fail.
    try {
      // Parses the incoming JSON request payload.
      const payload = await readBody(request);
      // Stores the completed assistant or feedback result.
      const result = await answerAssistantRequest(payload);
      // Continues the current operation.
      logRoutingDecision(result);
      // Sends the API response as JSON.
      sendJson(response, 200, result);
    // Handles an error from the preceding operation.
    } catch (error) {
      // Sends the API response as JSON.
      sendJson(response, 502, {
        // Sets this property in the current object.
        error: error.message,
        // Sets this property in the current object.
        mode: "ollama-error",
      });
    }
    // Continues the current operation.
    return;
  }

  // Sends the API response as JSON.
  sendJson(response, 404, {
    // Sets this property in the current object.
    error: "Not found",
  });
});

// Starts the local API server on the configured port.
server.listen(PORT, "127.0.0.1", () => {
  // Writes operational information to the terminal.
  console.log(`AMIDS assistant API running on http://127.0.0.1:${PORT}`);
  // Writes operational information to the terminal.
  console.log(`Using Ollama routing model ${OLLAMA_ROUTING_MODEL} at ${OLLAMA_URL}`);
  // Writes the specialised ranker configuration.
  console.log(`Routing provider ${routingProvider}; PyTorch ranker at ${PYTORCH_RANKER_URL}`);
  // Writes operational information to the terminal.
  console.log(
    // Assigns the computed value for the current operation.
    `AI-first routing: Qwen candidates <= ${qwenCandidateLimit}, cache TTL ${assistantCacheTtlMs}ms`
  );
});
