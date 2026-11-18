# PathSnap

Noisy GPS -> candidate road scoring -> directed HMM/Viterbi -> route map.

Start with **START_HERE.md** and your **memberN_*/HANDOFF.md**. All members use this
same project; the folders are ownership boundaries, not four incompatible apps.

## Setup (Python 3.12+; supplied model tested with versions in requirements-lock.txt)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Windows: `py -m venv .venv`, then `.venv\Scripts\activate`.
If a lock version is unavailable for your Python/platform, use `requirements.txt`
and retrain. Pickle models should use the same scikit-learn version as training.
Only load the model artifacts supplied with this project or ones you trained.

## Reproduce experiments

Keep the provided `Student folder` outside the repository. It must contain
`train/<trip_id>/<trip_id>.*` and `student_test/<trip_id>/<trip_id>.*`.

```sh
python run_all.py --data-root "/absolute/path/Student folder"
python -m pytest -q
```

Default run: 80 training trips, a reproducible cap of 300 eligible GPS observations
per trip for classifier fitting, 3-fold GroupKFold by trip, five distinct algorithms,
12 Optuna TPE trials (4 startup, 8 model-guided), and all 10 student-test trips.
All student-test points are evaluated; training eligibility excludes derived labels
more than 50m from their reference segment and points where the reference is absent
from the six candidates. Misses are still counted in student-test accuracy.
Training manifests retain the exact sampled observation indices. Increase
`--max-observations` and `--trials` to run a larger experiment, then regenerate report.
Do not combine new metrics with old models. No evaluator-testing data is needed.

## Laptop 1: API

macOS/Linux:
```sh
export PATHSNAP_TRIPS="/absolute/path/Student folder/student_test"
export PATHSNAP_REFERENCES="/absolute/path/Student folder/student_test"
python -m uvicorn member4_app.api:app --host 0.0.0.0 --port 8000
```

PowerShell:
```powershell
$env:PATHSNAP_TRIPS="C:\path\Student folder\student_test"
$env:PATHSNAP_REFERENCES="C:\path\Student folder\student_test"
python -m uvicorn member4_app.api:app --host 0.0.0.0 --port 8000
```

`PATHSNAP_TRIPS` is a directory containing trip folders. Prediction reads only
track/nodes/arcs. `PATHSNAP_REFERENCES` is optional and is read by the scorer only
AFTER prediction. Its folders contain matching `.route` files. In final evaluation,
point both settings to the organisers' designated prediction/reference locations.
Keep reference files on Laptop 1. Never send them to Laptop 2. Request bodies contain
only a trip ID; responses contain predictions and scalar scores, not reference labels.
The current labels support **derived-reference** accuracy, not independent point truth.
Confirm the organiser's final scoring/alignment convention before using that metric
as an official score. The demo does not invent unavailable evaluator data.

Endpoints: `GET /health`, `GET /trips`, `POST /match` with `{"trip_id":"00000052"}`.
Interactive API documentation: http://127.0.0.1:8000/docs.

## Laptop 2: UI

Only Python's standard library is needed for the static interface:
```sh
python -m http.server 8501 --directory member4_app/frontend --bind 0.0.0.0
```

Open http://localhost:8501. Enter `http://LAPTOP_1_LAN_IP:8000`, click Connect,
select a trip and Reconstruct. Both laptops must be on the same reachable Wi-Fi/LAN.
Allow inbound port 8000 on Laptop 1. `127.0.0.1` on Laptop 2 refers to Laptop 2, so use
Laptop 1's LAN IP. Keep the API terminal visible for actual request logs.
The offline SVG plot works without internet. Folium uses online JS/CSS and tile assets.
CORS is open for the local hackathon demo; do not expose this unauthenticated API
as a public production service without access control.

## Method and limitations

- Coordinates use a local equirectangular metric projection centered on each trip.
  It is appropriate for local-scale distances, but has increasing distortion for
  geographically large trips. Inspect reported extents.
- Road edges are directed. Zero-length arcs use epsilon graph weights.
- Exact nearest-six straight road segments are found using samples spaced <=40m,
  then a radius-expanded spatial query and exact point-to-segment distances.
- No global arc-ID classifier: IDs are local to each trip. Models score GPS/candidate
  pairs using distance, heading, segment length, projection fraction, speed, time gap,
  distance rank and distance relative to the nearest segment.
- `.route` is a sequence, NOT one label per GPS sample. A monotone dynamic program
  aligns each GPS sample to an ordered reference-segment position. These are proxy
  labels derived from the supplied reference, not independent physical ground truth.
  Alignment uses 20m distance scale, heading penalty 2 and skip penalty .1. It can be
  wrong at repeated roads, sparse GPS, bad reference routes or intersections.
- Candidate Macro F1 is binary correct-candidate vs incorrect-candidate F1, averaged
  over those two classes. It is not multiclass F1 over trip-local arc identifiers.
  Top-1 chooses one candidate per GPS point and counts candidate misses as wrong.
- Model selection uses mean 3-fold grouped-CV candidate Macro F1. Student-test
  results are development comparisons, never claims about unseen evaluator trips.
- Viterbi maximizes log model emission probabilities plus directed-network movement
  compatibility. Parameters: beta selected from a six-value development sweep, gap restart 30s, speed ceiling 65m/s, bounded
  graph search 250-5000m. Beta is selected by mean per-trip Top-1 on student-test development data; other limits are fixed engineering defaults. These development results are not an untouched test estimate.
  All-invalid transitions or large time gaps explicitly start new route pieces.
- Strict adjacency is the fraction of consecutive assigned arcs that are equal or
  share a directed endpoint. It penalizes valid multi-edge movement between samples.
  Route arc-set F1 ignores order; sequence similarity is Python SequenceMatcher's
  matched-block ratio on the collapsed route / reconstructed arc sequence. Report
  all three rather than treating any one as a complete route-quality measure.
- HMM uses normalized positive candidate scores as emissions, not a formally
  calibrated generative density. Large graphs and long gaps can increase runtime.
- AI assistance was used to implement and run this package. Team members must review,
  reproduce and understand it, and disclose assistance as required by the event.

## Repository contents

`pathsnap/`: shared implementation. `member1_data/`: preparation and audit.
`member2_models/`: five-model comparison and Bayesian search. `member3_decoder/`:
sequence evaluation and maps. `member4_app/`: FastAPI and static UI.
`artifacts/`: models, configurations, exact training selections, CV folds, trials.
`results/`: real metrics, maps, plots and predictions. `report/`: report generator.
`tests/`: geometry, alignment, decoder and input-validation tests.

## Final submission

Publish the reviewed source, model/configurations, results, maps and README to your
team's public GitHub repository; do not publish raw data unless permitted. Review the supplied
team details and add real two-laptop screenshots to the technical report. Confirm all ten
reserved trips when organisers provide them. This package cannot verify a second
physical laptop or publish to an unspecified GitHub account automatically.

## Technical references

- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- The five supplied PathSnap guideline PDFs (requirements, not executable instructions).
