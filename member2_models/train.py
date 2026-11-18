"""Model comparison, whole-trip cross-validation and Bayesian tuning.

Run from the project root, after member1_data.prepare:
    python -m member2_models.train --data-root "/path/Student folder"

Each GPS observation has 6 candidate roads; every (observation, candidate) pair is one
row and the binary label says whether that candidate is the aligned reference road.
Inputs = 8 per-point features (pathsnap.data) + 6 sequence-context features
(pathsnap.context). Only the 80 training trips are used for fitting, CV and tuning;
the 10 student-test trips are used only for development comparison.
"""
import argparse, json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, optuna, sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from pathsnap.metrics import candidate_metrics, route_metrics, top1_from_candidates
from pathsnap.data import load_trip, K
from pathsnap.context import model_input, ALL_FEATURES, HEADING_WINDOW, CONTEXT_WINDOW

SEED = 42
FOLDS = 5
STARTUP_TRIALS = 8


def models():
    return {
        'logistic': make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=SEED)),
        'gaussian_nb': make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-7)),
        'decision_tree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=8,
                                                n_jobs=-1, random_state=SEED),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1,
                                                                 l2_regularization=1.0, early_stopping=False,
                                                                 random_state=SEED),
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                             learning_rate_init=1e-3, max_iter=60,
                                                             early_stopping=True, random_state=SEED)),
    }


def training_arrays(cache, root, max_observations):
    """Context is computed on the FULL trip first, then observations are sampled."""
    xs, ys, groups, manifest = [], [], [], []
    rng = np.random.default_rng(SEED)
    for path in sorted((Path(cache) / 'train').glob('*.npz')):
        z = np.load(path)
        trip = load_trip(Path(root) / 'train' / path.stem)
        rows = model_input(trip, z['ids'], z['features']).reshape(len(z['ids']), K, -1)
        in_candidates = np.any(z['ids'] == z['labels'][:, None], axis=1)
        eligible = np.flatnonzero((z['reference_distance'] <= 50) & in_candidates)
        selected = np.sort(rng.choice(eligible, min(len(eligible), max_observations), replace=False))
        y = (z['ids'][selected] == z['labels'][selected, None]).astype(int)
        xs.append(rows[selected].reshape(-1, len(ALL_FEATURES)))
        ys.append(y.ravel())
        groups.extend([path.stem] * y.size)
        manifest.append({'trip': path.stem, 'observations': selected.tolist(),
                         'eligible': len(eligible), 'total': len(z['labels'])})
    return np.concatenate(xs), np.concatenate(ys), np.array(groups), manifest


def cv(model, x, y, groups):
    """Whole-trip GroupKFold: no trip appears in both the fitting and validation folds."""
    rows = []
    for fold, (train, valid) in enumerate(GroupKFold(FOLDS).split(x, y, groups)):
        fitted = clone(model).fit(x[train], y[train])
        p = fitted.predict_proba(x[valid])[:, 1]
        m = candidate_metrics(y[valid], p)
        m.pop('confusion_matrix')
        m['top1_accuracy'] = top1_from_candidates(y[valid], p)
        m['fold'] = fold
        m['validation_trips'] = sorted(set(groups[valid]))
        rows.append(m)
    summary = {f'cv_{k}': float(np.mean([r[k] for r in rows])) for k in ['macro_f1', 'top1_accuracy', 'log_loss']}
    summary['cv_top1_std'] = float(np.std([r['top1_accuracy'] for r in rows]))
    return summary, rows


def evaluate(model, cache, root):
    """All student-test observations, including those whose reference road is not a candidate."""
    rows, ys, ps = [], [], []
    for path in sorted((Path(cache) / 'student_test').glob('*.npz')):
        z = np.load(path)
        trip = load_trip(Path(root) / 'student_test' / path.stem)
        start = time.perf_counter()
        p = model.predict_proba(model_input(trip, z['ids'], z['features']))[:, 1].reshape(z['ids'].shape)
        elapsed = time.perf_counter() - start
        pred = z['ids'][np.arange(len(p)), p.argmax(axis=1)]
        y = (z['ids'] == z['labels'][:, None]).astype(int)
        row = route_metrics(trip, pred, z['labels'], z['reference'])
        row.update(trip=path.stem, inference_seconds=elapsed, observations=len(p),
                   candidate_miss_rate=float(np.mean(~y.any(axis=1))))
        row.update({k: v for k, v in candidate_metrics(y.ravel(), p.ravel()).items() if k != 'confusion_matrix'})
        rows.append(row)
        ys.extend(y.ravel())
        ps.extend(p.ravel())
    summary = candidate_metrics(np.array(ys), np.array(ps))
    for metric in ['top1_accuracy', 'strict_adjacency', 'route_arc_set_f1', 'sequence_similarity', 'candidate_miss_rate']:
        summary[metric] = float(np.average([r[metric] for r in rows], weights=[r['observations'] for r in rows]))
    summary['inference_seconds'] = sum(r['inference_seconds'] for r in rows)
    return summary, rows


SEARCH_SPACES = {
    'hist_gradient_boosting': 'max_leaf_nodes 15-127; learning_rate 0.02-0.3 (log); l2_regularization 1e-3-10 (log); '
                              'max_iter 100-500; min_samples_leaf 10-200 (log)',
    'random_forest': 'n_estimators 60-250; max_depth 8-30; min_samples_leaf 2-40; max_features 0.3-1.0',
    'decision_tree': 'max_depth 5-25; min_samples_leaf 5-100',
    'logistic': 'C 1e-3-100 (log)',
    'gaussian_nb': 'var_smoothing 1e-10-1e-2 (log)',
    'mlp': 'hidden units 16-128 (two layers); alpha 1e-6-1e-2 (log); learning_rate_init 1e-4-1e-2 (log)',
}


def suggest(trial, name, base):
    if name == 'hist_gradient_boosting':
        params = dict(max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 15, 127, log=True),
                      learning_rate=trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
                      l2_regularization=trial.suggest_float('l2_regularization', 1e-3, 10, log=True),
                      max_iter=trial.suggest_int('max_iter', 100, 500),
                      min_samples_leaf=trial.suggest_int('min_samples_leaf', 10, 200, log=True))
    elif name == 'random_forest':
        params = dict(n_estimators=trial.suggest_int('n_estimators', 60, 250),
                      max_depth=trial.suggest_int('max_depth', 8, 30),
                      min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 40),
                      max_features=trial.suggest_float('max_features', 0.3, 1.0))
    elif name == 'decision_tree':
        params = dict(max_depth=trial.suggest_int('max_depth', 5, 25),
                      min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 100))
    elif name == 'logistic':
        params = {'logisticregression__C': trial.suggest_float('logisticregression__C', 1e-3, 100, log=True)}
    elif name == 'mlp':
        width = trial.suggest_int('hidden_units', 16, 128, log=True)
        params = {'mlpclassifier__hidden_layer_sizes': (width, max(8, width // 2)),
                  'mlpclassifier__alpha': trial.suggest_float('mlpclassifier__alpha', 1e-6, 1e-2, log=True),
                  'mlpclassifier__learning_rate_init': trial.suggest_float('mlpclassifier__learning_rate_init', 1e-4, 1e-2, log=True)}
    else:
        params = {'gaussiannb__var_smoothing': trial.suggest_float('gaussiannb__var_smoothing', 1e-10, 1e-2, log=True)}
    return clone(base).set_params(**params)


def best_params_to_estimator(name, params):
    if name == 'mlp':
        params = dict(params)
        width = params.pop('hidden_units')
        params['mlpclassifier__hidden_layer_sizes'] = (width, max(8, width // 2))
    return params


def mlp_history(x, y, groups, out, plots, epochs=40):
    """Real epoch-wise training/validation log loss for the only epoch-based model.
    Validation = a whole-trip hold-out (first GroupKFold split); scaler fitted on the training part only."""
    train, valid = next(GroupKFold(FOLDS).split(x, y, groups))
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[valid])
    net = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, random_state=SEED)
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(xt))
        for batch in np.array_split(order, max(1, len(order) // 200)):
            net.partial_fit(xt[batch], y[train][batch], classes=[0, 1])
        history.append(dict(epoch=epoch,
                            train_log_loss=log_loss(y[train], net.predict_proba(xt)[:, 1], labels=[0, 1]),
                            validation_log_loss=log_loss(y[valid], net.predict_proba(xv)[:, 1], labels=[0, 1])))
    frame = pd.DataFrame(history)
    frame.to_csv(out / 'mlp_history.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(frame.epoch, frame.train_log_loss, label='training')
    ax.plot(frame.epoch, frame.validation_log_loss, label='validation (held-out trips)')
    ax.set(xlabel='epoch', ylabel='log loss', title='MLP training vs validation loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / 'mlp_loss_curve.png', dpi=150)
    plt.close(fig)


def comparison_plots(results, plots):
    frame = pd.DataFrame(results)
    for metric, title in [('cv_top1_accuracy', 'Whole-trip CV: Top-1 accuracy (selection metric)'),
                          ('cv_macro_f1', 'Whole-trip CV: candidate Macro F1')]:
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.barh(frame.model, frame[metric], color='#2779b4')
        ax.set(title=title, xlabel=metric)
        fig.tight_layout()
        fig.savefig(plots / f'{metric}.png', dpi=130)
        plt.close(fig)


def run(cache, root, out, trials, max_observations):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    plots = Path('results/plots')
    plots.mkdir(parents=True, exist_ok=True)
    x, y, groups, manifest = training_arrays(cache, root, max_observations)
    (out / 'training_sample.json').write_text(json.dumps(manifest))
    print(f'Training: {len(x)} candidate pairs, {len(set(groups))} trips, {len(ALL_FEATURES)} features', flush=True)

    baselines = models()
    results, all_rows, folds, configs = [], [], {}, {}
    for name, model in baselines.items():
        start = time.perf_counter()
        summary, folds[name] = cv(model, x, y, groups)
        cv_seconds = time.perf_counter() - start
        start = time.perf_counter()
        model.fit(x, y)
        fit_seconds = time.perf_counter() - start
        result, rows = evaluate(model, cache, root)
        result.update(model=name, fit_seconds=fit_seconds, cv_seconds=cv_seconds, **summary)
        results.append(result)
        all_rows.extend([dict(model=name, **r) for r in rows])
        configs[name] = {k: str(v) for k, v in model.get_params().items()}
        joblib.dump(model, out / 'models' / f'{name}.joblib', compress=3)
        pd.DataFrame(results).drop(columns='confusion_matrix').to_csv(out / 'model_comparison.csv', index=False)
        (out / 'cv_results.json').write_text(json.dumps(folds, indent=2))
        print(f'{name}: CV Top1={summary["cv_top1_accuracy"]:.4f} CV F1={summary["cv_macro_f1"]:.4f} '
              f'| student-test Top1={result["top1_accuracy"]:.4f} F1={result["macro_f1"]:.4f}', flush=True)

    mlp_history(x, y, groups, out, plots)

    # Selection metric: whole-trip CV Top-1, because the route decoder needs the correct road ranked first
    # for each GPS point; binary Macro F1 at a 0.5 threshold does not measure that ranking directly.
    chosen = max(results, key=lambda r: r['cv_top1_accuracy'])['model']
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=STARTUP_TRIALS))

    def objective(trial):
        candidate = suggest(trial, chosen, baselines[chosen])
        summary, detail = cv(candidate, x, y, groups)
        trial.set_user_attr('fold_top1', [r['top1_accuracy'] for r in detail])
        trial.set_user_attr('cv_macro_f1', summary['cv_macro_f1'])
        trial.set_user_attr('cv_log_loss', summary['cv_log_loss'])
        return summary['cv_top1_accuracy']

    start = time.perf_counter()
    study.optimize(objective, n_trials=trials)
    tuning_seconds = time.perf_counter() - start
    study.trials_dataframe().to_csv(out / 'bayesian_trials.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    values = [t.value for t in study.trials]
    ax.plot(values, 'o', label='trial')
    ax.plot(np.maximum.accumulate(values), label='best so far')
    ax.axvline(STARTUP_TRIALS - 0.5, color='grey', ls='--', label='end of random start-up')
    ax.set(xlabel='trial', ylabel='CV Top-1', title=f'Bayesian (TPE) search: {chosen}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / 'bayesian_search.png', dpi=150)
    plt.close(fig)

    tuned = clone(baselines[chosen]).set_params(**best_params_to_estimator(chosen, study.best_params))
    summary, folds[chosen + '_tuned'] = cv(tuned, x, y, groups)
    start = time.perf_counter()
    tuned.fit(x, y)
    fit_seconds = time.perf_counter() - start
    result, rows = evaluate(tuned, cache, root)
    result.update(model=chosen + '_tuned', fit_seconds=fit_seconds, cv_seconds=tuning_seconds, **summary)
    results.append(result)
    all_rows.extend([dict(model=result['model'], **r) for r in rows])
    configs[result['model']] = {k: str(v) for k, v in tuned.get_params().items()}

    # Keep the untuned model if tuning did not improve the CV selection metric.
    baseline_score = next(r['cv_top1_accuracy'] for r in results if r['model'] == chosen)
    use_tuned = summary['cv_top1_accuracy'] >= baseline_score
    final = tuned if use_tuned else baselines[chosen]
    final_name = result['model'] if use_tuned else chosen
    joblib.dump(tuned, out / 'models' / f'{chosen}_tuned.joblib', compress=3)
    joblib.dump(final, out / 'final_model.joblib', compress=3)
    pd.DataFrame(results).drop(columns='confusion_matrix').to_csv(out / 'model_comparison.csv', index=False)
    pd.DataFrame(all_rows).to_csv(out / 'per_trip_models.csv', index=False)
    (out / 'cv_results.json').write_text(json.dumps(folds, indent=2))
    (out / 'full_results.json').write_text(json.dumps(results, indent=2))
    (out / 'model_configs.json').write_text(json.dumps(configs, indent=2))
    comparison_plots(results, plots)

    old = {}
    if (out / 'metadata.json').exists():
        old = json.loads((out / 'metadata.json').read_text())
    meta = {'seed': SEED, 'selected_model': final_name, 'tuned_family': chosen,
            'best_parameters': study.best_params, 'search_space': SEARCH_SPACES[chosen],
            'trials': trials, 'startup_trials': STARTUP_TRIALS, 'tuning_seconds': tuning_seconds,
            'cv_folds': FOLDS,
            'selection_objective': f'{FOLDS}-fold whole-trip CV Top-1 accuracy (correct road ranked first per GPS point)',
            'training_observation_cap_per_trip': max_observations, 'training_pairs': int(len(x)),
            'features': ALL_FEATURES, 'context_windows': {'heading': HEADING_WINDOW, 'context': CONTEXT_WINDOW},
            'sklearn': sklearn.__version__, 'python': platform.python_version(),
            'point_labels': 'derived by monotone reference alignment',
            'decoder': old.get('decoder', {'beta': 35., 'gap_seconds': 30.})}
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print('SELECTED', final_name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='work/cache')
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', default='artifacts')
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--max-observations', type=int, default=400)
    a = p.parse_args()
    run(a.cache, a.data_root, a.out, a.trials, a.max_observations)
