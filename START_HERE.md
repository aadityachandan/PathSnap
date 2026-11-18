# Start here: four-member handoff

## What is provided

A shared source project, member-owned modules, reproducible experiments, saved model
artifacts, development results, maps, a FastAPI backend, a static UI, and a generated
technical report. Consult VALIDATION.md for exactly what was actually run.

Give every member the full PathSnap.zip. Do not send only one source folder: the
modules import shared code. Assign ownership as follows:

| Member | Read first | Owns |
|---|---|---|
| Aaditya Shankar Chandan (PES1UG24CS003) | member1_data/HANDOFF.md | Raw data, reference alignment, candidates, features, EDA |
| Aaditya Vashisht (PES1UG24CS004) | member2_models/HANDOFF.md | Five models, grouped CV, Bayesian tuning, saved model |
| Ramya H R (PES1UG25CS835) | member3_decoder/HANDOFF.md | Directed graph, HMM/Viterbi, sequence evaluation |
| Vinayaka Jagadish (PES1UG25CS855) | member4_app/HANDOFF.md | API, UI, two-laptop integration, submission assembly |

## Do these together

1. Unzip the project and read README.md. Everyone uses the same project version.
2. Each person reviews and explains their component; the package is AI-assisted.
3. Install dependencies on the training/backend laptop. The UI laptop only needs Python.
4. Start the supplied backend and UI using README commands. First test on one laptop.
5. Put the backend and UI on separate laptops on the same network. Connect by LAN IP.
6. Run the ten student-test trips and inspect the maps and terminal request logs.
7. Capture real UI and receiving-laptop terminal screenshots. Verify the included team names/IDs and add
   those screenshots to the report. Generated local evidence does not prove a two-laptop demo.
8. Ask the organiser which alignment/scoring convention to use for official pointwise
   accuracy and when/how the ten evaluator trips will be supplied. Do not train on them.
9. Publish the reviewed code/results/model to your team's public GitHub repository.
10. Each person explains their part in the viva, including real limitations and failures.

## Rebuilding the report after adding evidence

Put your real screenshots in `report/evidence/` as PNG files. Put team information in
`report/team.txt`. The generator includes those files when present:

```sh
python -m report.build --data-root "/path/Student folder"
```

Do not call the report final until your team has reviewed it and added the actual
required two-laptop evidence. No metrics should be typed in manually.

## Simplest demo commands

Laptop 1 (inside the activated environment):

```sh
python member4_app/start_backend.py --trips "/path/Student folder/student_test" --references "/path/Student folder/student_test"
```

Laptop 2:

```sh
python member4_app/start_ui.py
```

Open http://localhost:8501 on Laptop 2 and enter Laptop 1 LAN IP in the API field.
