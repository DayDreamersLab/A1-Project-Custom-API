# AMIDS PyTorch Route Ranker

This service is a small trainable AI model specialised for two decisions:

1. Which approved `routeRegistry` entries best match a user's request?
2. Does the user want one route or multiple related routes?

It is not a generative chatbot. It scores all 128 approved routes in one
PyTorch batch and normally returns only IDs, confidence, and scope. The Node
gateway remains the security boundary: it validates returned IDs and retrieves
the authoritative paths from `src/data/routeRegistry.js`.

## Architecture

```text
React UI
  -> Node AMIDS assistant gateway :3001
     -> PyTorch route ranker :8001
        -> relevance model scores all approved routes
        -> scope model predicts single or multiple
     -> Node validates IDs against routeRegistry
     -> hybrid mode sends uncertain requests to Qwen
```

The ranker uses hashed word/character n-gram features and a small neural
pair-ranking network. This makes it trainable, CPU-friendly, and independent of
internet model downloads. Accuracy comes from the quality and coverage of its
expert-reviewed training examples.

## 1. Create The Python Environment

Python 3.11 is a conservative choice for the Windows workplace machine.

Windows PowerShell:

```powershell
py -3.11 -m venv pytorch_route_ranker\.venv
pytorch_route_ranker\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r pytorch_route_ranker\requirements.txt
```

macOS/Linux:

```bash
python3 -m venv pytorch_route_ranker/.venv
source pytorch_route_ranker/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r pytorch_route_ranker/requirements.txt
```

For an offline workplace network, download approved Python wheels on an
approved connected machine, scan them, move them through the organisation's
approved transfer process, and install from an internal wheel directory:

```powershell
python -m pip install --no-index --find-links C:\approved-wheelhouse -r pytorch_route_ranker\requirements.txt
```

## 2. Export The Approved Registry

Run this whenever `src/data/routeRegistry.js` changes:

```bash
npm run ranker:export
```

The model refuses to start when its checkpoint was trained against a different
registry version. This prevents stale models from returning removed route IDs.

## 3. Prepare Training Examples

Generate initial examples from route titles, descriptions, keywords, topics,
and route purposes:

```bash
npm run ranker:generate
```

Add expert-reviewed cases to:

```text
pytorch_route_ranker/data/expert_training_examples.jsonl
```

Each line has this shape:

```json
{"query":"show all runway wind data","scope":"multiple","relevantRouteIds":["wind-overview","wind-current-observations"],"source":"expert-reviewed"}
```

Use the exact routes an aviation expert expects. Add difficult phrases,
paraphrases, abbreviations, exclusions, and examples where similar wording
should produce different results. Do not train automatically from a merely
`helpful` click; collect a corrected expected route selection first.

## 4. Train And Evaluate

```bash
npm run ranker:train
npm run ranker:evaluate
```

Training writes the ignored local checkpoint:

```text
pytorch_route_ranker/models/route_ranker.pt
```

Evaluation reports:

- single/multiple scope accuracy;
- top-route accuracy;
- relevant-route recall;
- fallback rate;
- average local latency.

Before production use, create a separate expert-reviewed test file that is
never used for training and run:

```bash
python -m pytorch_route_ranker.scripts.evaluate --data path/to/held_out_test.jsonl
```

## 5. Start The Ranker And Gateway

Terminal 1:

```bash
npm run ranker:api
```

Terminal 2:

```bash
AMIDS_ROUTING_PROVIDER=hybrid npm run api
```

Terminal 3:

```bash
npm run dev
```

Routing modes:

- `hybrid`: use PyTorch first and send uncertain/unavailable cases to Qwen.
- `pytorch`: use PyTorch and emergency registry fallback; never call Qwen.
- `ollama`: preserve the previous Qwen-only behavior.

Windows PowerShell example:

```powershell
$env:AMIDS_ROUTING_PROVIDER="hybrid"
npm run api
```

## API

Health:

```text
GET http://127.0.0.1:8001/health
```

Rank:

```http
POST http://127.0.0.1:8001/rank
Content-Type: application/json

{
  "query": "show all runway wind data",
  "roleKey": "pilot",
  "maxRoutes": 8,
  "routeBiases": {
    "wind-current-observations": 0.5
  }
}
```

The response includes `routeId`, `routeIds`, `requestScope`, `confidence`,
`needsFallback`, and diagnostic scores. Although the Python response includes
route metadata for testing, the Node gateway ignores its paths and resolves
the selected IDs from its own approved registry.

## Important Limitations

- The initial generated examples teach the registry's existing vocabulary; they
  do not automatically teach every aviation synonym or operational nuance.
- Expert examples are the main mechanism for specialisation.
- Confidence thresholds must be calibrated against held-out workplace queries.
- Keep Qwen fallback enabled until accuracy and recall meet an agreed test
  threshold.
- This is a navigation aid. It must not make flight-safety or operational
  decisions on behalf of pilots, ATC, or dispatchers.
