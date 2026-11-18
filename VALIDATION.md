# Validation and remaining external steps

Completed locally (second development iteration):
- Parsed and audited all 80 training and 10 student-test trips.
- Error analysis of the first classifier: most mistakes were opposite-direction twin roads
  and adjacent segments. Added six sequence-context features; window sizes chosen by
  5-fold whole-trip CV on training trips only (artifacts/context_window_sweep.csv).
- Trained six model families with five whole-trip validation folds; selection metric is
  CV Top-1. MLP epoch-wise training/validation loss recorded (artifacts/mlp_history.csv).
- Ran 30 Bayesian TPE trials (8 start-up) on hist_gradient_boosting and retained all trial records.
- Found and fixed a decoder issue: small backward GPS slips on the same road were treated
  as impossible moves, causing false route breaks. Swept beta x backward tolerance on
  student-test development data (selected beta=10.0, tolerance=40.0m).
- Evaluated classifier and HMM on all 19,160 student-test GPS observations.
- Seven automated tests passed.
- All ten student-test trips returned HTTP 200 from a real local uvicorn server
  (results/logs/api.log) and via FastAPI TestClient (results/api_validation.json).

Weighted derived-reference Top-1: classifier 0.8548; HMM 0.8926.
(First iteration: classifier 0.7838; HMM 0.7608.) These are development measurements
using aligned proxy labels, not official evaluator scores.

Note: the six baseline model files in artifacts/models were re-saved after the run by
refitting with identical data and seeds; each reproduced its recorded student-test Top-1.

Not completed or claimed:
- A two-physical-laptop network demonstration and its receiving-terminal screenshots.
- Evaluation on the ten reserved evaluator trips (not supplied).
- Independent review, reproduction and viva preparation by each named team member.
