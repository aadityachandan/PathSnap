# Validation and remaining external steps

Completed locally:
- Parsed and audited all 80 training and 10 student-test trips.
- Trained five model families with three whole-trip validation folds.
- Ran 12 Bayesian TPE trials and retained all trial records.
- Ran a six-value decoder beta sweep using student-test development data.
- Evaluated classifier and HMM on all 19,160 student-test GPS observations.
- Seven automated tests passed, including reference-free inference and scoring isolation.
- All ten student-test trips returned HTTP status 200 via FastAPI TestClient (in process).
- A real local browser connected over HTTP and reconstructed trip 00000052.
- Browser-tested the offline plot and Folium view; corrected hidden-iframe map sizing.

Weighted derived-reference Top-1: classifier 0.7838; HMM 0.7608. These are development
measurements using aligned proxy labels, not official evaluator scores. HMM can
reduce accuracy on some trips; the report includes those regressions.

Not completed or claimed:
- A two-physical-laptop network demonstration and its receiving-terminal screenshots.
- Evaluation on the ten reserved evaluator trips (not supplied).
- Publication to your team's GitHub account.
- Independent review, reproduction and viva preparation by each named team member.

The supplied report is a reviewed development report, not proof those external steps
have happened. Follow START_HERE.md and TEAM_CHECKLIST.md before submission.
