# API, UI and integration

Assigned to: **VINAYAKA JAGADISH (PES1UG25CS855)**

You receive the FULL PathSnap project. Your owned files are `member4_app/api.py, member4_app/frontend/index.html, pathsnap/maps.py`.

## First steps

1. Read START_HERE.md and install dependencies using README.md.
2. Read your owned code and its docstrings.
3. Run from the project root:

```sh
python -m uvicorn member4_app.api:app --host 0.0.0.0 --port 8000
```

## What you personally need to do

Set PATHSNAP_TRIPS and optional PATHSNAP_REFERENCES as in README. Run backend on Laptop 1 and static UI on Laptop 2. Use Laptop 1 LAN IP in the UI. Demonstrate requests in terminal, offline plot, Folium map, errors and JSON export.

## Handoff

Collect REAL two-laptop screenshots; assemble final report and publish the reviewed team repository. Do not expose reference labels in API responses.

## Viva preparation

Prepare a two-minute explanation of your component, one real result, one failure case and one limitation. Run `python -m pytest -q` after changes. Do not claim authorship or experiments you cannot explain; disclose AI assistance as required.

## Run the integrated version

From the repository root, with dependencies installed:

```sh
python member4_app/start_backend.py --trips "/path/to/Student folder/student_test"
```

In a second terminal (or on Laptop 2):

```sh
python member4_app/start_ui.py --port 8501
```

Open http://localhost:8501 and connect to http://127.0.0.1:8000 for a local check,
or Laptop 1's LAN address for the two-laptop demonstration. Select a trip, run
matching, inspect both map views, and download the JSON. Add `--references`
with the development trip directory only when you want optional scoring.

This integration uses the repository's existing model and 14-feature pipeline.
Validation results are recorded in results/api_validation.json and results/logs.
The supplied report still requires genuine two-laptop screenshots before final submission.
