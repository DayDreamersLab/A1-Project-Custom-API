import { useState } from "react";
import {
  askAssistant,
  createFeedbackRecord,
  submitAssistantFeedback,
} from "../services/assistantEngine";
import {
  queryRequestsAllRelated,
  queryRequestsNavigation,
} from "../services/queryIntent";

function writeInitialReservedNavigationMessage(reservedWindow) {
  try {
    reservedWindow.document.open();
    reservedWindow.document.write(`
      <!doctype html>
      <title>Opening AMIDS route...</title>
      <body style="font-family: system-ui, sans-serif; padding: 24px;">
        <p></p>
      </body>
    `);
    reservedWindow.document.querySelector("p").textContent =
      "currently matching your request to an AMIDS route";
    reservedWindow.document.close();
  } catch {
    // Some browsers restrict writing to the reserved tab. Navigation can still work.
  }
}

function reserveNavigationWindow() {
  const reservedWindow = window.open("", "_blank");
  if (!reservedWindow) return null;

  writeInitialReservedNavigationMessage(reservedWindow);

  return reservedWindow;
}

function closeReservedWindow(reservedWindow) {
  try {
    if (reservedWindow && !reservedWindow.closed) {
      reservedWindow.close();
    }
  } catch {
    // Ignore close failures; the user can close the tab manually.
  }
}

function getRecommendedRoutes(result) {
  if (!result) return [];
  if (Array.isArray(result.routes) && result.routes.length > 0) {
    return result.routes;
  }
  return result.route ? [result.route] : [];
}

function getRecommendationKey(result, query, roleKey) {
  if (!result) return null;
  if (result.recommendationId) return result.recommendationId;

  const routeIds = getRecommendedRoutes(result)
    .map((route) => route.id)
    .filter(Boolean)
    .sort()
    .join(",");

  return `${roleKey}::${(result.requestQuery ?? query).trim().toLowerCase()}::${routeIds}`;
}

export default function AssistantPanel({
  userId,
  roleKey,
  preferences,
  userProfile,
  expertRules,
  onAssistantResult,
  onOpenLink,
  onShowRoutes,
  onProfileUpdated,
}) {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [feedbackLog, setFeedbackLog] = useState([]);
  const [feedbackDecisions, setFeedbackDecisions] = useState(() => new Map());
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [notice, setNotice] = useState("");

  const recommendedRoutes = getRecommendedRoutes(result);
  const openTarget = recommendedRoutes[0] ?? null;
  const hasMultipleRoutes = recommendedRoutes.length > 1;
  const requiresClarification = Boolean(result?.requiresClarification);
  const shouldShowRouteList = requiresClarification || hasMultipleRoutes;
  const currentRecommendationKey = getRecommendationKey(result, query, roleKey);
  const currentFeedbackDecision = currentRecommendationKey
    ? feedbackDecisions.get(currentRecommendationKey)
    : null;
  const hasRatedCurrentRecommendation =
    Boolean(currentFeedbackDecision);

  async function handleAsk(event) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    const wantsAllRelatedRoutes = queryRequestsAllRelated(trimmedQuery);
    const reservedWindow =
      queryRequestsNavigation(trimmedQuery) && !wantsAllRelatedRoutes
        ? reserveNavigationWindow()
        : null;

    setIsLoading(true);
    setNotice("");

    try {
      const assistantResult = await askAssistant({
        query: trimmedQuery,
        userId,
        roleKey,
        preferences,
        userProfile,
        expertRules,
      });

      setResult(assistantResult);
      onAssistantResult(assistantResult);

      const notices = [];

      if (assistantResult.backendError) {
        notices.push(
          assistantResult.requiresClarification
            ? `The routing services could not confirm a route: ${assistantResult.backendError}`
            : `Assistant API issue: ${assistantResult.backendError}`
        );
      }

      const assistantRoutes = getRecommendedRoutes(assistantResult);
      const shouldAutoOpenSingleRoute =
        assistantResult.shouldOpen &&
        assistantResult.openMode !== "multiple" &&
        assistantRoutes.length === 1;

      if (shouldAutoOpenSingleRoute) {
        const routeToOpen = assistantRoutes[0];
        const routeDescription = routeToOpen.description || routeToOpen.title || "the selected routeRegistry page";
        const opened = onOpenLink(routeToOpen, {
          targetWindow: reservedWindow,
          loadingMessage: `currently opening matched AMIDS route containing ${routeDescription}`,
          redirectDelayMs: 2500,
        });
        if (!opened) {
          closeReservedWindow(reservedWindow);
          notices.push("The assistant found a route, but the browser blocked auto-open. Use the open button below.");
        }
      } else {
        closeReservedWindow(reservedWindow);
        if (assistantResult.shouldOpen && assistantRoutes.length > 1) {
          notices.push("The assistant found multiple related routes. Review them, then display them on this screen.");
        }
      }

      if (notices.length > 0) {
        setNotice(notices.join(" "));
      }
    } catch (error) {
      closeReservedWindow(reservedWindow);
      setNotice(error.message);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleFeedback(rating) {
    const recommendationKey = getRecommendationKey(result, query, roleKey);
    if (
      !result ||
      !recommendationKey ||
      isSubmittingFeedback ||
      feedbackDecisions.has(recommendationKey)
    ) {
      return;
    }

    const feedbackRecord = createFeedbackRecord({
      query,
      userId,
      roleKey,
      result,
      rating,
    });

    setFeedbackDecisions((currentDecisions) => {
      const nextDecisions = new Map(currentDecisions);
      nextDecisions.set(recommendationKey, rating);
      return nextDecisions;
    });
    setIsSubmittingFeedback(true);

    try {
      const response = await submitAssistantFeedback(feedbackRecord);

      if (response.ok && response.profile) {
        const savedFeedbackRecord = response.feedbackRecord ?? feedbackRecord;
        setFeedbackLog((currentLog) => [savedFeedbackRecord, ...currentLog].slice(0, 3));
        if (response.duplicate && response.feedbackRecord?.rating) {
          setFeedbackDecisions((currentDecisions) => {
            const nextDecisions = new Map(currentDecisions);
            nextDecisions.set(recommendationKey, response.feedbackRecord.rating);
            return nextDecisions;
          });
        }
        onProfileUpdated(response.profile);
        setNotice(response.duplicate
          ? "Feedback was already recorded for this recommendation. Personalization was not changed again."
          : `Feedback saved. Personalization updated for ${(response.profile.preferredRouteIds ?? []).length} preferred route(s).`);
      } else if (response.error) {
        setFeedbackDecisions((currentDecisions) => {
          const nextDecisions = new Map(currentDecisions);
          nextDecisions.delete(recommendationKey);
          return nextDecisions;
        });
        setNotice(`Feedback was not saved, so you can try again: ${response.error}`);
      }
    } finally {
      setIsSubmittingFeedback(false);
    }
  }

  function handleShowAll() {
    const displayedCount = onShowRoutes(recommendedRoutes);
    if (displayedCount < recommendedRoutes.length) {
      setNotice(
        `Displayed ${displayedCount} of ${recommendedRoutes.length} routes. Some route paths were unavailable or invalid.`
      );
    }
  }

  return (
    <section className="assistant-panel" aria-labelledby="assistant-title">
      <div className="assistant-panel-header">
        <div>
          <p className="eyebrow">AI-Ready Layer</p>
          <h3 id="assistant-title">AMIDS Assistant</h3>
        </div>
        <span className="assistant-mode">{result?.mode ?? "route-registry-ready"}</span>
      </div>

      <form className="assistant-query" onSubmit={handleAsk}>
        <label htmlFor="assistant-query">Ask for the information you need</label>
        <div>
          <input
            id="assistant-query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Example: Give me runway wind data"
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? "Checking..." : "Ask"}
          </button>
        </div>
      </form>

      {userProfile && (
        <section className="personalization-summary" aria-label="Personalization summary">
          <h4>Personalization active</h4>
          <p>
            Preferred routes:{" "}
            {(userProfile.preferredRouteIds ?? []).length > 0
              ? userProfile.preferredRouteIds.slice(0, 3).join(", ")
              : "none yet"}
          </p>
          <p>
            Frequent topics:{" "}
            {(userProfile.frequentTopics ?? []).length > 0
              ? userProfile.frequentTopics.slice(0, 4).join(", ")
              : "none yet"}
          </p>
        </section>
      )}

      {result && (
        <article className={`assistant-result${requiresClarification ? " assistant-result-uncertain" : ""}`}>
          <h4>{requiresClarification ? "Clarification needed" : "Recommendation"}</h4>
          <p>{result.explanation}</p>
          {requiresClarification && (
            <div className="assistant-clarification" role="status">
              <strong>{result.clarificationPrompt}</strong>
              <p>Select a possible route below, or refine your request and ask again.</p>
            </div>
          )}
          {notice && <p>{notice}</p>}
          {(result.appliedRules ?? []).length > 0 && (
            <p>Applied rule: {result.appliedRules.join(", ")}</p>
          )}
          {shouldShowRouteList ? (
            <div className="assistant-route-list">
              <div className="assistant-route-list-topline">
                <h5>
                  {requiresClarification
                    ? `${recommendedRoutes.length} possible route${recommendedRoutes.length === 1 ? "" : "s"}`
                    : `${recommendedRoutes.length} related routes found`}
                </h5>
                {!requiresClarification && (
                  <button className="open-link-button" type="button" onClick={handleShowAll}>
                    Display all on this screen
                  </button>
                )}
              </div>
              {recommendedRoutes.map((route) => (
                <article className="assistant-route-card" key={route.id}>
                  <div>
                    <strong>{route.title}</strong>
                    <p>{route.description}</p>
                  </div>
                  <button type="button" onClick={() => onOpenLink(route)}>
                    {requiresClarification ? "Choose and open" : "Open"}
                  </button>
                </article>
              ))}
            </div>
          ) : (
            openTarget && (
              <button
                className="open-link-button"
                type="button"
                onClick={() => onOpenLink(openTarget)}
              >
                Open {openTarget.title ?? openTarget.label}
              </button>
            )
          )}
          {!requiresClarification && (
            <div className="assistant-feedback">
              <button
                type="button"
                disabled={isSubmittingFeedback || hasRatedCurrentRecommendation}
                onClick={() => handleFeedback("helpful")}
              >
                {isSubmittingFeedback
                  ? "Saving..."
                  : currentFeedbackDecision === "helpful"
                    ? "Helpful recorded"
                    : "Helpful"}
              </button>
              <button
                type="button"
                disabled={isSubmittingFeedback || hasRatedCurrentRecommendation}
                onClick={() => handleFeedback("not-helpful")}
              >
                {currentFeedbackDecision === "not-helpful"
                  ? "Not helpful recorded"
                  : "Not helpful"}
              </button>
            </div>
          )}
        </article>
      )}

      {feedbackLog.length > 0 && (
        <div className="feedback-log">
          <h4>Feedback records ready for backend logging</h4>
          {feedbackLog.map((entry) => (
            <p key={entry.id}>
              {entry.rating} feedback stored for {entry.recommendedRouteId ?? "no route"}.
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
