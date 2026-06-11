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

The generator deliberately creates several language styles:

```text
Command:   open Radar Satellite Composite
Fragment:  Radar Satellite Composite
Polite:    Radar Satellite Composite please
Search:    looking for Radar Satellite Composite
Question:  where can I find Radar Satellite Composite
Broad:     radar data
Broad:     everything related to radar
```

Bare shared topics such as `radar` are labelled as multiple-route requests,
while bare route titles and sufficiently distinguishing keywords are labelled
as single-route requests. The generator removes a query entirely if two
generation rules assign it contradictory expected answers.

Generated examples include a `source` value such as
`generated-route-fragment` or `generated-topic-command`. These labels make it
possible to audit whether the dataset is overly dominated by one language
style. Do not automatically generate misspellings or highly ambiguous natural
requests; add those only when their intended routes can be labelled
confidently.

Add expert-reviewed cases to:

```text
pytorch_route_ranker/data/expert_training_examples.jsonl
```

Each line has this shape:

```json
{"query":"show all runway wind data","scope":"multiple","relevantRouteIds":["wind-overview","wind-current-observations"],"source":"expert-reviewed"}
```

Real AMIDS route IDs do not need to follow the prototype's suffix convention.
For example, `radar-satellite-composite` is valid. The generator always creates
single-route examples from the route title and keywords. It creates synthetic
purpose-wide examples only when route metadata confidently matches a known
shared purpose.

Place the route's most useful broad grouping term first in `keywords`. For
example:

```json
{
  "id": "radar-satellite-composite",
  "title": "Radar Satellite Composite",
  "path": "https://internal-amids.example/radar-composite",
  "description": "Combined radar and satellite imagery.",
  "keywords": ["radar", "satellite", "composite imagery"]
}
```

Here, generated broad queries such as `show all radar data` can group it with
other routes whose first keyword is also `radar`. Review broad generated groups
carefully and describe operationally important groupings explicitly in
`expert_training_examples.jsonl`.

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

## Automated Reproducible Experiments

Use the experiment runner instead of manually creating folders for every
training attempt:

```bash
npm run ranker:experiment -- \
  --run-name radar-keyword-update \
  --notes "Added radar synonyms and clarified composite descriptions" \
  --held-out-test pytorch_route_ranker/data/held_out_test.jsonl
```

Windows PowerShell uses the same command on one line:

```powershell
npm run ranker:experiment -- --run-name radar-keyword-update --notes "Added radar synonyms" --held-out-test pytorch_route_ranker\data\held_out_test.jsonl
```

The command automatically:

1. exports the current approved registry;
2. regenerates synthetic training examples;
3. creates a timestamped immutable run folder;
4. snapshots training data, held-out test data, and model source code;
5. records Python, Node, Git, registry, and training configuration;
6. trains a new model inside the run folder;
7. evaluates that exact model against the snapshotted held-out test;
8. stores all logs and appends metrics to `pytorch_route_ranker/runs/summary.csv`.

Each completed run resembles:

```text
pytorch_route_ranker/runs/20260610-143000-radar-keyword-update/
  configuration.json
  notes.txt
  model.pt
  training-log.txt
  evaluation-log.txt
  data/
  source/
```

Run folders are ignored by Git because they may contain large models and
confidential registry snapshots. Back them up using an approved internal
location. The runner never automatically promotes a model to
`pytorch_route_ranker/models/route_ranker.pt`; compare `summary.csv` and review
mismatches before promotion.

All npm ranker commands use `scripts/runPythonModule.mjs` to find `py -3.11`,
`python`, or `python3`, making the same commands usable on Windows and macOS.
Set `AMIDS_PYTHON_COMMAND` if the workplace Python executable uses another
name.

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
