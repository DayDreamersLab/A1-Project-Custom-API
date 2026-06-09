const navigationIntentWords = new Set([
  "open",
  "go",
  "goto",
  "navigate",
  "take",
  "show",
  "display",
  "view",
  "load",
  "bring",
  "get",
  "give",
  "need",
  "find",
  "list",
]);

const broadInformationWords = new Set([
  "data",
  "information",
  "info",
  "route",
  "routes",
  "page",
  "pages",
  "site",
  "sites",
  "website",
  "websites",
  "source",
  "sources",
  "option",
  "options",
  "result",
  "results",
  "product",
  "products",
  "report",
  "reports",
  "detail",
  "details",
  "set",
  "collection",
  "weather",
]);

const explicitBroadPhrases = [
  "all related",
  "everything about",
  "everything related",
  "all available",
  "every available",
  "each available",
  "any and all",
  "complete set",
  "full set",
  "entire set",
  "whole set",
];

export function normalize(text) {
  return String(text ?? "").toLowerCase().trim();
}

export function tokenize(text) {
  return normalize(text).split(/[^a-z0-9/]+/).filter(Boolean);
}

export function queryRequestsNavigation(query) {
  return tokenize(query).some((token) => navigationIntentWords.has(token));
}

export function queryRequestsAllRelated(query) {
  const queryText = normalize(query);
  const tokens = tokenize(queryText);

  if (
    !queryText ||
    /\b(?:not|don't|do not|without)(?:\s+[a-z0-9/]+){0,3}\s+(?:all|every|everything|each|complete|full|comprehensive|entire|whole)\b/.test(queryText)
  ) {
    return false;
  }

  if (explicitBroadPhrases.some((phrase) => queryText.includes(phrase))) {
    return true;
  }

  const hasNavigationIntent = tokens.some((token) => navigationIntentWords.has(token));
  const hasInformationWord = tokens.some((token) => broadInformationWords.has(token));
  const hasAll = tokens.includes("all");
  const hasEvery =
    tokens.includes("every") || tokens.includes("everything") || tokens.includes("each");
  const hasCompletenessWord = tokens.some((token) =>
    ["complete", "full", "comprehensive", "entire", "whole"].includes(token)
  );

  if (hasEvery) return tokens.length > 1;
  if (hasAll) return hasNavigationIntent || hasInformationWord || tokens.length >= 3;
  return hasCompletenessWord && (hasNavigationIntent || hasInformationWord);
}

export function queryRequestsOpen(query) {
  return queryRequestsAllRelated(query) || queryRequestsNavigation(query);
}
