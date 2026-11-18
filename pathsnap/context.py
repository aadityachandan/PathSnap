"""Sequence-context features added on top of the eight per-point features.

The eight base features (pathsnap.data.FEATURES) describe one GPS point in isolation.
Error analysis on student-test showed that most classifier mistakes are
(a) picking the opposite-direction twin of the correct road, because the
one-second heading is noisy, and (b) picking an adjacent segment at segment
boundaries. These features use neighbouring GPS points to fix that.
They use only track/nodes/arcs and candidate IDs; never the reference route.
"""
import numpy as np
from pathsnap.data import FEATURES

CONTEXT_FEATURES = [
    'heading_cos_smooth',     # heading over a +/-10 point window, less noisy than 1-step heading
    'window_presence',        # fraction of the +/-10 neighbouring points that also have this arc as a candidate
    'window_mean_distance_m', # mean distance from those neighbours to this arc
    'window_nearest_frac',    # fraction of neighbours for which this arc is the nearest candidate
    'connects_prev',          # arc equals or directly follows a candidate of the previous point
    'connects_next',          # arc equals or directly precedes a candidate of the next point
]
ALL_FEATURES = FEATURES + CONTEXT_FEATURES
HEADING_WINDOW = 10
CONTEXT_WINDOW = 10


def _shift(array, offset, fill):
    out = np.full_like(array, fill)
    if offset > 0:
        out[:-offset] = array[offset:]
    elif offset < 0:
        out[-offset:] = array[:offset]
    else:
        out[:] = array
    return out


def context_features(trip, ids, features):
    n, k = ids.shape
    arcs = trip.arcs
    # 1. smoothed heading
    w = HEADING_WINDOW
    ahead = trip.gps[np.minimum(np.arange(n) + w, n - 1)]
    behind = trip.gps[np.maximum(np.arange(n) - w, 0)]
    motion = ahead - behind
    norm = np.linalg.norm(motion, axis=1)
    direction = motion / np.maximum(norm[:, None], 1e-9)
    seg = trip.xy[arcs[ids, 1]] - trip.xy[arcs[ids, 0]]
    seg = seg / np.maximum(np.linalg.norm(seg, axis=2, keepdims=True), 1e-9)
    heading = np.sum(seg * direction[:, None, :], axis=2)
    heading[norm < 3.0] = 0.0          # (almost) stationary: direction unknown

    # 2. agreement with neighbouring points' candidate sets
    distance = features[:, :, 0]
    present = np.zeros((n, k)); dist_sum = np.zeros((n, k)); nearest = np.zeros((n, k)); valid = np.zeros((n, 1))
    for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
        if offset == 0:
            continue
        other_ids = _shift(ids, offset, -1)
        other_dist = _shift(distance, offset, np.inf)
        exists = (other_ids[:, :1] >= 0).astype(float)
        match = ids[:, :, None] == other_ids[:, None, :]           # (n, k, k)
        hit = match.any(axis=2)
        matched_dist = np.where(match, other_dist[:, None, :], np.inf).min(axis=2)
        present += hit
        dist_sum += np.where(hit, matched_dist, 0.0)
        nearest += ids == other_ids[:, :1]
        valid += exists
    valid = np.maximum(valid, 1)
    presence = present / valid
    mean_distance = np.where(present > 0, dist_sum / np.maximum(present, 1), 100.0)
    nearest_frac = nearest / valid

    # 3. road-network connectivity with previous / next point's candidates
    def connects(other):
        same = ids[:, :, None] == other[:, None, :]
        ok = other[:, None, :] >= 0
        return same, ok
    prev_ids = _shift(ids, -1, -1); next_ids = _shift(ids, 1, -1)
    same, ok = connects(prev_ids)
    follows = arcs[np.maximum(prev_ids, 0), 1][:, None, :] == arcs[ids, 0][:, :, None]
    connects_prev = ((same | follows) & ok).any(axis=2)
    same, ok = connects(next_ids)
    precedes = arcs[ids, 1][:, :, None] == arcs[np.maximum(next_ids, 0), 0][:, None, :]
    connects_next = ((same | precedes) & ok).any(axis=2)
    connects_prev[0] = True; connects_next[-1] = True

    return np.stack([heading, presence, mean_distance, nearest_frac,
                     connects_prev, connects_next], axis=-1).astype(np.float32)


def model_input(trip, ids, features):
    """(n, k, 8) base features -> (n*k, 14) rows in the order the classifier expects."""
    full = np.concatenate([features, context_features(trip, ids, features)], axis=-1)
    return full.reshape(-1, len(ALL_FEATURES))
