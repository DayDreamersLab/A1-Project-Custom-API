import { routeRegistry } from "../data/routeRegistry";
import {
  normalize,
  queryRequestsAllRelated,
  queryRequestsOpen,
  tokenize,
} from "./queryIntent";

const maxRelatedRoutes = 8;

function toPublicRoute(route) {
  return {
    id: route.id,
    title: route.title,
    path: route.path,
    description: route.description,
  };
}

function createLocalRecommendationId(query, roleKey, routes) {
  return [
    "local",
    roleKey ?? "unknown-role",
    normalize(query),
    routes.map((route) => route.id).filter(Boolean).sort().join(","),
  ].join("::");
}

function scoreRoute(route, query, roleKey, preferences, expertRules, userProfile = {}) {
  const queryText = normalize(query);
  const queryTokens = tokenize(query);
  const searchableText = normalize(
    [
      route.title,
      route.description,
      ...route.keywords,
    ].join(" ")
  );

  const exactKeywordScore = route.keywords.reduce(
    (score, keyword) => score + (queryText.includes(normalize(keyword)) ? 4 : 0),
    0
  );
  const queryScore = queryTokens.reduce(
    (score, token) => {
      const singularToken = token.endsWith("s") ? token.slice(0, -1) : token;
      return score + (searchableText.includes(token) || searchableText.includes(singularToken) ? 2 : 0);
    },
    0
  );
  const preferenceScore =
    preferences.preferredCategories.some((category) => searchableText.includes(category)) ||
    preferences.preferredLinks.includes(route.id)
      ? 3
      : 0;
  const roleScore = route.keywords.includes(roleKey) ? 5 : 0;
  const profileRouteScore = Math.max(-6, Math.min(6, userProfile.routeScores?.[route.id] ?? 0));
  const preferredRouteScore = userProfile.preferredRouteIds?.includes(route.id) ? 6 : 0;
  const avoidedRouteScore = userProfile.avoidedRouteIds?.includes(route.id) ? -6 : 0;
  const frequentTopicScore = (userProfile.frequentTopics ?? []).reduce((score, topic) => {
    return searchableText.includes(normalize(topic)) ? score + 2 : score;
  }, 0);
  const expertRuleScore = expertRules.reduce((score, rule) => {
    const triggerMatches = queryText.includes(normalize(rule.trigger));
    const linkMatches = rule.linkIds.includes(route.id);
    const instructionMatches = tokenize(rule.instruction ?? "").some((token) =>
      searchableText.includes(token)
    );

    if (!triggerMatches) return score;
    if (linkMatches) return score + 8;
    if (rule.linkIds.length === 0 && instructionMatches) return score + 4;

    return score;
  }, 0);

  return (
    exactKeywordScore +
    queryScore +
    preferenceScore +
    roleScore +
    profileRouteScore +
    preferredRouteScore +
    avoidedRouteScore +
    frequentTopicScore +
    expertRuleScore
  );
}

function buildLocalRecommendation({ query, roleKey, preferences, expertRules, userProfile }) {
  const shouldReturnMultipleRoutes = queryRequestsAllRelated(query);
  const scoredRoutes = routeRegistry
    .map((route) => ({
      ...route,
      score: scoreRoute(route, query, roleKey, preferences, expertRules, userProfile),
    }))
    .filter((route) => route.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, shouldReturnMultipleRoutes ? maxRelatedRoutes : 1);

  const selectedRoute = scoredRoutes[0] ?? null;
  const publicRoutes = scoredRoutes.map(toPublicRoute);

  return {
    recommendationId: createLocalRecommendationId(query, roleKey, publicRoutes),
    requestQuery: query,
    mode: "local-route-registry",
    roleKey,
    recommendedLinkIds: [],
    openLinkId: null,
    route: selectedRoute ? toPublicRoute(selectedRoute) : null,
    routes: publicRoutes,
    openMode: publicRoutes.length > 1 ? "multiple" : "single",
    shouldOpen: queryRequestsOpen(query) && Boolean(selectedRoute),
    explanation: selectedRoute
      ? shouldReturnMultipleRoutes
        ? `Matched ${publicRoutes.length} related routes against routeRegistry using local keyword scoring. Start the Ollama API to let Qwen 3 rank registry candidates.`
        : "Matched against routeRegistry using local keyword scoring. Start the Ollama API to let Qwen 3 rank registry candidates."
      : "No strong route-registry match was found. Add registry keywords or start the Ollama API for better interpretation.",
    appliedRules: expertRules
      .filter((rule) => normalize(query).includes(normalize(rule.trigger)))
      .map((rule) => rule.name),
  };
}

async function fetchWithTimeout(url, options, timeoutMs = 20000) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } finally {
    window.clearTimeout(timeout);
  }
}

async function createAssistantApiError(response) {
  try {
    const payload = await response.json();
    const detail = payload.error ? `: ${payload.error}` : "";
    return new Error(`Assistant API returned ${response.status}${detail}`);
  } catch {
    return new Error(`Assistant API returned ${response.status}`);
  }
}

function describeAssistantApiError(error, assistantApiUrl) {
  if (error.name === "AbortError") {
    return `Assistant API request timed out at ${assistantApiUrl}`;
  }

  if (error instanceof TypeError) {
    return `Could not connect to assistant API at ${assistantApiUrl}. Confirm npm run api is running and CORS allows this frontend origin.`;
  }

  return error.message;
}

function getAssistantApiBaseUrl() {
  return (
    import.meta.env.VITE_AMIDS_ASSISTANT_API_BASE_URL ??
    "http://127.0.0.1:3001/api/amids-assistant"
  );
}

function getAssistantRouteUrl() {
  return import.meta.env.VITE_AMIDS_ASSISTANT_API_URL ?? getAssistantApiBaseUrl();
}

async function readJsonResponse(response) {
  if (!response.ok) {
    throw await createAssistantApiError(response);
  }

  return response.json();
}

export async function askAssistant(payload) {
  const assistantApiUrl = getAssistantRouteUrl();

  try {
    const response = await fetchWithTimeout(assistantApiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    return readJsonResponse(response);
  } catch (error) {
    console.warn("Falling back to route-registry assistant engine:", error);
    return {
      ...buildLocalRecommendation(payload),
      backendError: describeAssistantApiError(error, assistantApiUrl),
    };
  }
}

export async function loadUserProfile({ userId, roleKey }) {
  const profileUrl = new URL(`${getAssistantApiBaseUrl()}/profile`);
  profileUrl.searchParams.set("userId", userId);
  profileUrl.searchParams.set("roleKey", roleKey);

  try {
    const response = await fetchWithTimeout(profileUrl.href, { method: "GET" }, 5000);
    return readJsonResponse(response);
  } catch (error) {
    console.warn("Could not load personalization profile:", error);
    return {
      ok: false,
      profile: null,
      error: describeAssistantApiError(error, profileUrl.href),
    };
  }
}

export async function submitAssistantFeedback(feedbackRecord) {
  const feedbackUrl = `${getAssistantApiBaseUrl()}/feedback`;

  try {
    const response = await fetchWithTimeout(feedbackUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(feedbackRecord),
    }, 10000);
    return readJsonResponse(response);
  } catch (error) {
    console.warn("Could not submit assistant feedback:", error);
    return {
      ok: false,
      error: describeAssistantApiError(error, feedbackUrl),
    };
  }
}

export function createFeedbackRecord({ query, roleKey, result, rating, userId }) {
  const recommendationQuery = result.requestQuery ?? query;

  return {
    id: crypto.randomUUID(),
    recommendationId: result.recommendationId,
    userId,
    query: recommendationQuery,
    roleKey,
    rating,
    recommendedLinkIds: result.recommendedLinkIds,
    recommendedRouteId: result.route?.id ?? null,
    recommendedRouteIds: result.routes?.map((route) => route.id) ?? [],
    result: {
      route: result.route ?? null,
      routes: result.routes ?? [],
      recommendationId: result.recommendationId,
      requestQuery: recommendationQuery,
      mode: result.mode,
      explanation: result.explanation,
    },
    timestamp: new Date().toISOString(),
    // Send this record to an internal feedback/audit API in the real AMIDS version.
  };
}
