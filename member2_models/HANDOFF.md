# Models and tuning

Assigned to: **AADITYA VASHISHT (PES1UG24CS004)**

You receive the FULL PathSnap project. Your owned files are `member2_models/train.py, pathsnap/metrics.py`.

## First steps

1. Read START_HERE.md and install dependencies using README.md.
2. Read your owned code and its docstrings.
3. Run from the project root:

```sh
python -m member2_models.train --data-root "/path/Student folder" --trials 30
```

## What you personally need to do

Explain all six models, the six context features and why they were added (error analysis), binary candidate labels, grouped cross-validation, the selection objective, TPE startup vs guided trials, sampling and tuned-versus-untuned results. Reproduce the actual winning configuration.

## Handoff

Give final_model.joblib and metadata.json to Members 3 and 4. Keep all six models, trial logs and result tables.

## Viva preparation

Prepare a two-minute explanation of your component, one real result, one failure case and one limitation. Run `python -m pytest -q` after changes. Do not claim authorship or experiments you cannot explain; disclose AI assistance as required.
