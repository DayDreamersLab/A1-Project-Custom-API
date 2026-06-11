# AI Implementation Notes

This prototype now supports a secure local Node gateway with either Qwen,
a specialised trainable PyTorch ranker, or a hybrid of both.

## Current Prototype Behavior

- `src/components/AssistantPanel.jsx` provides the user-facing assistant UI.
- `src/services/assistantEngine.js` contains the current local rule/search engine.
- `src/data/assistantConfig.js` contains example personalization and expert rules.
- `src/data/roleData.js` contains approved navigation targets with placeholder `href: "#"` values.
- `src/data/routeRegistry.js` contains 128 exemplar AMIDS simulation routes used by the local Ollama API.
- AI recommendations highlight matching link cards in `DetailPanel`.

## Where The AI API Goes

The prototype now calls a local assistant API from `src/services/assistantEngine.js`:

```js
const response = await fetch("http://127.0.0.1:3001/api/amids-assistant", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});
return response.json();
```

For this prototype, that endpoint is implemented in
`server/ollamaAssistantServer.mjs`. In the default `hybrid` mode, it asks the
specialised PyTorch model first and escalates uncertain requests to local
Ollama Qwen 3. The returned route paths always come directly from the Node
gateway's `routeRegistry`; Python and Qwen select approved IDs only. The local
API serves `/amids/routes/...` paths for simulation. Do not call either model
directly from the browser.

## Backend Responsibilities

The future `/api/amids-assistant` endpoint should:

1. Authenticate the user.
2. Load the user's role, permissions, preferences, and approved expert rules.
3. Retrieve only approved AMIDS navigation targets.
4. Apply safety and organization rules before personalization.
5. Call the AI model with structured navigation options.
6. Require structured JSON output.
7. Return a validated route object whose `path` comes directly from `routeRegistry`.
8. Store feedback/audit records internally.

## Expected AI Response Shape

```json
{
  "mode": "ai-api",
  "roleKey": "pilot",
  "openMode": "single",
  "route": {
    "id": "wind-runway-impact",
    "title": "Wind Runway Impact",
    "path": "/amids/routes/wind-runway-impact",
    "description": "Wind Runway Impact: runway-specific operational impact and threshold checks for wind, gust, crosswind, tailwind, runway wind, and wind shear information."
  },
  "routes": [
    {
      "id": "wind-runway-impact",
      "title": "Wind Runway Impact",
      "path": "/amids/routes/wind-runway-impact",
      "description": "Wind Runway Impact: runway-specific operational impact and threshold checks for wind, gust, crosswind, tailwind, runway wind, and wind shear information."
    }
  ],
  "shouldOpen": true,
  "explanation": "The request matched a routeRegistry route for runway-specific wind information.",
  "appliedRules": ["Runway wind means wind shear first"]
}
```

## Safety Rules To Enforce

- Recommend approved AMIDS sources only.
- Never make final operational decisions.
- Never hide mandatory safety warnings.
- Always preserve source links and timestamps when real data is used.
- Let official AMIDS safety rules override user personalization.
- Treat expert instructions as explicit rules only after user/admin approval.

## Personalization Vs Expert Rules

Personalization should adjust ordering and defaults:

- usual airport
- default role
- preferred categories
- frequent links

When routing is uncertain, the prototype lets the user select one or several
suggested routes. One fixed `+0.5` route-evidence budget is divided across the
selected routes, and the selected count is recorded as a corrected `single` or
`multiple` request scope. `None of these match` is retained as review/training
evidence but does not alter route or topic preference scores. Each
clarification result can update personalization only once.

Expert rules should act as explicit workflow instructions:

- "When I ask for runway wind data, show wind shear first."
- "When I ask about LVP, show low visibility procedure and RVR information."

Safety and organization rules should override both.

## Current AI Boundary

The specialised PyTorch service is responsible for:

- scoring all approved routes for relevance;
- predicting whether the request needs one or multiple routes;
- returning confidence so uncertain cases can be escalated;
- learning task-specific language from expert-reviewed examples.

Qwen remains responsible for uncertain or unsupported requests in `hybrid`
mode. Set `AMIDS_ROUTING_PROVIDER` to `pytorch`, `hybrid`, or `ollama` to
control this behavior.

Normal deterministic code is responsible only for:

- validating that model-selected IDs exist in the approved registry;
- retrieving the selected paths from `routeRegistry`;
- caching repeated responses and coalescing duplicate requests;
- applying explicit user scope constraints and bounded preference adjustments;
- returning selectable route suggestions when the configured AI is uncertain or unavailable.

See `pytorch_route_ranker/README.md` for setup, training, evaluation, and API
instructions.
