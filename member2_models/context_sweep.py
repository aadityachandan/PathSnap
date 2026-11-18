"""Choose the sequence-context window sizes with whole-trip CV on TRAINING trips only.

    python -m member2_models.context_sweep --data-root "/path/Student folder"

Stage 1 varies the heading window with the context window fixed; stage 2 varies the
context window at the best heading window. A fixed HistGradientBoosting model is used
so that only the feature windows change. Results: artifacts/context_window_sweep.csv.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
import pathsnap.context as context
from pathsnap.data import load_trip, K
from pathsnap.metrics import top1_from_candidates

SEED = 42


def cv_top1(trips, heading, window, max_observations):
    context.HEADING_WINDOW, context.CONTEXT_WINDOW = heading, window
    rng = np.random.default_rng(SEED)
    xs, ys, groups = [], [], []
    for trip_id, z, trip in trips:
        rows = context.model_input(trip, z['ids'], z['features']).reshape(len(z['ids']), K, -1)
        y = z['ids'] == z['labels'][:, None]
        eligible = np.flatnonzero((z['reference_distance'] <= 50) & y.any(axis=1))
        sel = np.sort(rng.choice(eligible, min(len(eligible), max_observations), replace=False))
        xs.append(rows[sel].reshape(-1, rows.shape[-1]))
        ys.append(y[sel].ravel().astype(int))
        groups += [trip_id] * y[sel].size
    x, y, groups = np.concatenate(xs), np.concatenate(ys), np.array(groups)
    scores = []
    for train, valid in GroupKFold(5).split(x, y, groups):
        model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=31, random_state=SEED).fit(x[train], y[train])
        scores.append(top1_from_candidates(y[valid], model.predict_proba(x[valid])[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))


def run(cache, root, max_observations):
    trips = [(p.stem, np.load(p), load_trip(Path(root) / 'train' / p.stem))
             for p in sorted((Path(cache) / 'train').glob('*.npz'))]
    records = []

    def record(stage, heading, window):
        mean, std = cv_top1(trips, heading, window, max_observations)
        records.append(dict(stage=stage, heading_window=heading, context_window=window, cv_top1=mean, cv_top1_std=std))
        print(f'heading={heading:2d} context={window:2d}: CV Top-1 {mean:.4f} (+/- {std:.4f})', flush=True)
        return mean

    best_heading = max([1, 3, 5, 10, 15], key=lambda h: record('heading', h, 10))
    for window in [0, 5, 15]:
        record('context', best_heading, window)
    frame = pd.DataFrame(records)
    Path('artifacts').mkdir(exist_ok=True)
    frame.to_csv('artifacts/context_window_sweep.csv', index=False)
    best = frame.loc[frame.cv_top1.idxmax()]
    print(f'BEST heading={int(best.heading_window)} context={int(best.context_window)} CV Top-1={best.cv_top1:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='work/cache')
    p.add_argument('--data-root', required=True)
    p.add_argument('--max-observations', type=int, default=400)
    a = p.parse_args()
    run(a.cache, a.data_root, a.max_observations)
