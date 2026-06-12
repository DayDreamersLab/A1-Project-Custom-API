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

### Reviewable Interaction Evidence

Every new clarification response is automatically exported to:

```text
pytorch_route_ranker/data/reviewable_interaction_evidence.jsonl
```

The file is a pending-review queue, not training data. A selected route is
stored as a proposed label that must be verified. A `none-match` response is
stored without proposed relevant routes so a reviewer can assign the correct
answer. User IDs are deliberately excluded, and the file is ignored by Git
because queries may be confidential.

Run this command to backfill or rebuild missing queue entries from the current
personalization store:

```bash
npm run ranker:export-evidence
```

Each record includes `reviewStatus`, `reviewNotes`, `approvedScope`, and
`approvedRelevantRouteIds` review fields. The automatic exporter preserves
those fields on later exports. Change `reviewStatus` from `pending` to
`approved` only after verifying the query, scope, and routes. Approved records
can later be converted into training examples through a separate controlled
import step.

### Generate Hard Examples From Approved Corrections

Approved interaction corrections can be expanded into a small, controlled set
of semantically equivalent training queries using a local Ollama model:

```bash
npm run ranker:export
npm run ranker:export-evidence
npm run ranker:generate-hard-examples
```

The first command refreshes `route_registry.json` from the actual
`routeRegistry.js`; the generator therefore uses the IDs, titles,
descriptions, and keywords from the active AMIDS registry. Route paths are
retained in the local registry but are never sent to the LLM.

The command reads only evidence records whose `reviewStatus` is `approved` and
whose `approvedScope` and `approvedRelevantRouteIds` are valid. The local LLM
is allowed to write paraphrases, but it is never allowed to choose or alter the
approved labels.

For each approved correction, the pipeline:

1. retains the original real failed query as an
   `approved-interaction-correction`;
2. asks the local LLM for 30 varied candidate paraphrases;
3. rejects candidates that change explicit single/multiple scope, lose
   important time or exclusion qualifiers, contain URLs or internal-style
   route IDs, or duplicate existing queries;
4. asks the local LLM a second time to strictly verify semantic equivalence;
5. retains at most 15 accepted `synthetic-hard-example` paraphrases.

Select a local model through an environment variable:

```bash
HARD_EXAMPLE_LLM_MODEL=qwen3:8b npm run ranker:generate-hard-examples
```

Optionally use a stronger local model for the semantic review pass:

```bash
HARD_EXAMPLE_LLM_MODEL=qwen3:1.7b \
HARD_EXAMPLE_VALIDATOR_MODEL=qwen3:8b \
npm run ranker:generate-hard-examples
```

Windows PowerShell:

```powershell
$env:HARD_EXAMPLE_LLM_MODEL="qwen3:8b"
npm run ranker:generate-hard-examples
```

Inspect the approved evidence and planned work without calling Ollama or
writing files:

```bash
npm run ranker:generate-hard-examples -- --dry-run
```

Process one correction or adjust the controlled limits:

```bash
npm run ranker:generate-hard-examples -- \
  --evidence-id correction-id \
  --generate-count 30 \
  --max-paraphrases 15
```

The resulting files are:

```text
pytorch_route_ranker/data/hard_example_training_data.jsonl
pytorch_route_ranker/data/hard_example_training_manifest.json
```

Both are ignored by Git because they may contain confidential user queries.
The manifest records generation counts, rejected candidates, registry
fingerprint, model, and local inference timings. The generated JSONL dataset is
automatically included by `ranker:train` and snapshotted by
`ranker:experiment`.

Do not normally use `--skip-semantic-validation`. It exists for controlled
diagnostics, but omitting the second review pass increases the risk that a
paraphrase changes the intended route, scope, location, time meaning, or
exclusion. If an approval is revoked or the route registry fingerprint
changes, the associated generated hard examples are removed on the next run.

## 4. Train And Evaluate

```bash
npm run ranker:train
npm run ranker:evaluate
```

Training uses `--device auto` by default. It selects CUDA when available,
otherwise Apple MPS when available, and otherwise CPU. Select a device
explicitly when needed:

```bash
npm run ranker:train -- --device cuda
npm run ranker:train -- --device cuda:1
npm run ranker:train -- --device mps
npm run ranker:train -- --device cpu
```

The model, route vectors, training targets, validation targets, loss weights,
and batch indices are moved onto the selected device. Checkpoints store the
training-device description but save model weights on CPU so they remain
portable to the lower-powered inference machine.

Training also displays and records both model-size counts:

```text
totalParameters=1639682 trainableParameters=1639682
```

`totalParameters` counts every model parameter. `trainableParameters` counts
only parameters whose `requires_grad` value allows the optimizer to update
them. Experiment configurations and `summary.csv` preserve both values, while
`npm run ranker:status` displays them for the active checkpoint.

Training writes the ignored local checkpoint:

```text
pytorch_route_ranker/models/route_ranker.pt
```

Evaluation reports:

- single/multiple scope accuracy;
- top-route accuracy;
- relevant-route recall;
- exact route-set accuracy, precision, F1, and route-count error;
- a dedicated breakdown for multiple-route requests;
- per-category metrics and a separate critical-example breakdown;
- fallback rate;
- average local latency.

Held-out examples can identify one or more categories and safety-critical cases:

```json
{"query":"show all runway wind data","scope":"multiple","relevantRouteIds":["wind-current-observations","wind-runway-impact"],"categories":["wind","runway"],"critical":true}
```

When `categories` is omitted, evaluation infers categories from the first
keyword of each expected route. Explicit test categories are preferred because
they remain meaningful if route metadata changes.

Before production use, create a separate expert-reviewed test file that is
never used for training and run:

```bash
npm run ranker:evaluate -- \
  --data path/to/held_out_test.jsonl \
  --output path/to/evaluation-results.json
```

## Automated Reproducible Experiments

Use the experiment runner instead of manually creating folders for every
training attempt:

```bash
npm run ranker:experiment -- \
  --run-name radar-keyword-update \
  --notes "Added radar synonyms and clarified composite descriptions" \
  --held-out-test pytorch_route_ranker/data/held_out_test.jsonl \
  --device cuda
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
8. stores detailed metrics in `evaluation-results.json`;
9. automatically compares the candidate with the active model when one exists;
10. stores `comparison.json`, all logs, and summary metrics in
    `pytorch_route_ranker/runs/summary.csv`.

Each completed run resembles:

```text
pytorch_route_ranker/runs/20260610-143000-radar-keyword-update/
  configuration.json
  notes.txt
  model.pt
  training-log.txt
  evaluation-log.txt
  evaluation-results.json
  comparison.json
  comparison-log.txt
  data/
  source/
```

Run folders are ignored by Git because they may contain large models and
confidential registry snapshots. Back them up using an approved internal
location. The runner never automatically promotes a model to
`pytorch_route_ranker/models/route_ranker.pt`; compare `summary.csv` and review
mismatches before promotion. A completed experiment never promotes itself.

## Compare, Promote, And Roll Back Models

Compare any candidate against the currently active checkpoint using the same
registry and frozen held-out test:

```bash
npm run ranker:compare -- \
  --candidate pytorch_route_ranker/runs/RUN_ID/model.pt \
  --data pytorch_route_ranker/data/held_out_test.jsonl \
  --output pytorch_route_ranker/runs/RUN_ID/comparison.json
```

The comparison rejects candidates that regress overall top-route or scope
accuracy, multiple-route recall or exact-set accuracy, critical examples, or
sufficiently represented categories. It also enforces fallback-rate and
latency limits. Review `comparison.json` even when the candidate is eligible.

Promote only an eligible, reviewed candidate:

```bash
npm run ranker:promote -- \
  --candidate pytorch_route_ranker/runs/RUN_ID/model.pt \
  --comparison pytorch_route_ranker/runs/RUN_ID/comparison.json \
  --approved-by "reviewer-name" \
  --reason "Improved held-out multiple-route recall without regressions"
```

For the first managed release only, when no active checkpoint exists, bootstrap
from a reviewed detailed evaluation:

```bash
npm run ranker:promote -- \
  --candidate pytorch_route_ranker/runs/RUN_ID/model.pt \
  --allow-initial \
  --initial-evaluation pytorch_route_ranker/runs/RUN_ID/evaluation-results.json \
  --approved-by "reviewer-name" \
  --reason "Approved initial managed model"
```

Promotion rejects stale comparisons, changed model files, and registry
mismatches. It archives immutable release copies, atomically replaces the
active checkpoint, and records the approval in
`pytorch_route_ranker/models/promotion_history.jsonl`. Check status and
available release IDs with:

```bash
npm run ranker:status
```

Roll back to an archived release:

```bash
npm run ranker:rollback -- \
  --release RELEASE_ID \
  --approved-by "reviewer-name" \
  --reason "Regression observed during controlled operational testing"
```

Restart `npm run ranker:api` after promotion or rollback. The running process
keeps its currently loaded checkpoint until restarted. These commands record
the supplied reviewer identity; production approval still requires appropriate
operating-system permissions and organisational access controls.

All npm ranker commands use `scripts/runPythonModule.mjs`. It prefers the
project's `pytorch_route_ranker/.venv`, then searches for `python`, `py -3.11`,
or `python3`, making the same commands usable on Windows and macOS. Set
`AMIDS_PYTHON_COMMAND` if the workplace Python executable uses another name.

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
`needsFallback`, `fallbackReasons`, and diagnostic scores. `fallbackReasons`
distinguishes low route confidence from uncertain single/multiple scope,
missing selections, and insufficient multiple-route selections. Although the Python response includes
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
