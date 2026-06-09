# Local Ollama API Setup

This prototype can use Qwen alone or as the fallback behind a specialised local
PyTorch ranker:

```text
User query
  -> specialised PyTorch ranker attempts a fast learned decision
  -> uncertain requests fall back to Qwen in hybrid mode
  -> validate selected IDs against routeRegistry
  -> cache the validated response for repeated requests
```

The Node gateway remains authoritative for every URL. See
`pytorch_route_ranker/README.md` to train and start the specialised model.

## 1. Install Ollama And Pull Qwen 3

Install Ollama on the local machine, then pull a small routing model:

```bash
ollama pull qwen3:0.6b
```

Other small models worth testing:

```bash
ollama pull gemma3:1b
ollama pull gemma3:270m
ollama pull llama3.2:1b
```

If you use a different model tag, start the API server with
`OLLAMA_ROUTING_MODEL` set. The older `OLLAMA_MODEL` setting remains supported
as a fallback.

## 2. Start The Local Ollama Runtime

Ollama usually runs automatically. To test:

```bash
ollama run qwen3:0.6b
```

Exit the chat after confirming the model works.

## 3. Start The AMIDS Assistant API

In one terminal:

```bash
npm run api
```

By default this starts:

```text
http://127.0.0.1:3001/api/amids-assistant
```

Optional model override:

```bash
OLLAMA_ROUTING_MODEL=qwen3:0.6b npm run api
```

Routing provider:

```bash
AMIDS_ROUTING_PROVIDER=hybrid npm run api
```

- `hybrid` tries the PyTorch ranker, then calls Qwen when confidence is low.
- `pytorch` never calls Qwen.
- `ollama` preserves Qwen-only behavior.

The default remains `qwen3:latest` so existing installations continue to work.
For the older CPU-only workstation, explicitly selecting `qwen3:0.6b` or
`gemma3:1b` should reduce ambiguous-query latency.

Optional routing and Ollama tuning:

```bash
AMIDS_QWEN_CANDIDATE_LIMIT=12 \
AMIDS_BROAD_QWEN_CANDIDATE_LIMIT=16 \
AMIDS_ASSISTANT_CACHE_TTL_MS=300000 \
AMIDS_ASSISTANT_CACHE_MAX_ENTRIES=200 \
OLLAMA_KEEP_ALIVE=-1 \
OLLAMA_NUM_CTX=2048 \
OLLAMA_BROAD_NUM_CTX=3072 \
OLLAMA_NUM_PREDICT=160 \
OLLAMA_ROUTING_MODEL=qwen3:0.6b \
npm run api
```

The API sends Ollama only each candidate's `id`, `title`, short purpose, and six
query-relevant keywords, plus compact relevant expert rules and personalization
hints. It never sends route paths or the full personalization profile.

The default shortlist contains 12 candidates. Reducing it makes prompt
evaluation faster but increases the chance that the correct route never reaches
Qwen. Increasing it improves recall but costs more prompt-processing time.

Explicit `all`, `every`, `everything`, or `each` requests use the broader
candidate and context limits. The API constrains the model response to
`requestScope: "multiple"` and requires multiple approved route IDs. The model
still chooses the routes; deterministic code only enforces the user's explicit
scope and provides an emergency fallback if the model violates it.

Repeated identical requests are served from a five-minute in-memory cache.
Concurrent identical requests share the same in-flight Qwen call. Feedback
updates clear the cache so new preferences can take effect.

If Ollama fails or returns prose instead of the required JSON, the API returns
the strongest deterministic `routeRegistry` match rather than failing the
entire request with a `502`.

## 4. Start The React Prototype

In another terminal:

```bash
npm run dev
```

Open the Vite URL, usually:

```text
http://localhost:5173
```

## 5. Try A Navigation Command

Select `Pilot`, then ask:

```text
Open runway wind data
```

The assistant should:

1. Send the request to the local Node API.
2. The Node API retrieves a compact shortlist from the 128-route `routeRegistry`.
3. Qwen interprets the request and decides whether it needs one route or multiple routes.
4. Qwen selects only approved IDs from the shortlist.
5. The API validates the IDs and retrieves their paths from `routeRegistry`.
6. The UI opens or offers to open the corresponding route or routes.

## 6. Measure Routing Speed

Each assistant response includes `routingDiagnostics`, and the API terminal
prints a compact routing line without printing the confidential query:

```text
[AMIDS routing] strategy=ai-qwen-route-selection qwen=yes cache=miss total=7895ms ...
```

When Qwen is used, a second line reports Ollama timing:

```text
[Ollama timing] roundTrip=7895ms load=0ms prompt=4200ms generation=3400ms overhead=295ms ...
```

- `roundTrip` is the complete local `fetch` call to Ollama. It includes model
  queueing, loading, prompt evaluation, generation, and response handling, not merely network/request transmission time.
- High `load`: confirm `OLLAMA_KEEP_ALIVE=-1`.
- High `prompt`: reduce `AMIDS_QWEN_CANDIDATE_LIMIT` or shorten registry keywords.
- High `generation`: use a smaller model or reduce `OLLAMA_NUM_PREDICT`.
- High `overhead`: check whether Ollama is handling another request or whether
  the machine is under heavy memory/CPU pressure.

A repeated request should instead show:

```text
[AMIDS routing] strategy=ai-response-cache qwen=no cache=hit total=1ms ...
```

The health endpoint reports the active tuning values:

```text
http://127.0.0.1:3001/api/health
```

The registry lives here:

```text
src/data/routeRegistry.js
```

Each route has:

```js
{
  id,
  title,
  path,
  description,
  keywords
}
```

For the local prototype, each registry path looks like:

```text
/amids/routes/wind-runway-impact
```

The local Node API serves these route paths directly. In a real AMIDS
environment, replace those `path` values with the actual internal AMIDS URLs.

## 7. Where Real AMIDS Links Go

The navigation assistant opens only the paths stored in:

```text
src/data/routeRegistry.js
```

Replace prototype paths like:

```js
/amids/routes/wind-runway-impact
```

with real internal AMIDS links only inside the secure workplace environment.
`roleData.js` remains separate and supplies the manual role/category buttons.

## 8. Important Security Shape

Use:

```text
React UI -> local Node API -> local Ollama
```

Avoid:

```text
React UI -> external AI API
```

The local Node API is where AMIDS can later enforce access control, safety rules, logging, and output validation.

## 9. Is This Already A Custom AI API?

Yes. `server/ollamaAssistantServer.mjs` is the application's custom AI API.
Ollama is the model runtime behind it. Rewriting the HTTP endpoint in another
language would remove very little of a `7895ms` round trip because most of that
time is model inference.

For lower first-request latency, keep this API contract and optimize or replace
the inference runtime behind it. Practical options include:

- use a smaller quantized routing model;
- keep the model loaded and prevent parallel duplicate requests;
- use `llama.cpp` or an optimized CPU inference runtime instead of Ollama;
- later add semantic embedding retrieval so Qwen sees fewer, higher-quality candidates.

Training a model from scratch is not required for route selection. The first
priority should be measuring `load`, `prompt`, `generation`, and `overhead`,
then optimizing whichever number dominates.
