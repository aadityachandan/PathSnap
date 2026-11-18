"""Exploratory data analysis for the raw PathSnap dataset.

Run from the project root, after member1_data.prepare:

    python -m member1_data.eda --data-root "/path/Student folder"

Reads the raw train/ and student_test/ trips and the inventory written by prepare\n(work/cache/inventory.csv, falling back to results/inventory.csv).
It never modifies raw data, the cache or any model input. Outputs go to results/eda/.
Pattern analysis uses the TRAINING partition; partition-level tables show both
train and student_test side by side. Evaluator-testing data is not used.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium

from pathsnap.data import load_trip

SPLITS = ["train", "student_test"]
FILES = {
    "track": ["lon", "lat", "time_s"],
    "nodes": ["lon", "lat"],
    "arcs": ["from_node", "to_node"],
    "route": ["arc_id"],
}


def trip_folders(root, split):
    return sorted(f for f in (root / split).iterdir() if f.is_dir())


def read_raw(path, columns):
    """Read a header-less whitespace file without validation, so problems stay visible."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python")


# ---------------------------------------------------------------- tables
def file_inventory_and_quality(root):
    """One row per (trip, file): availability, size, rows, columns, dtype and invalid values."""
    file_rows, quality_rows = [], []
    for split in SPLITS:
        for folder in trip_folders(root, split):
            tid = folder.name
            frames = {}
            for ext, cols in FILES.items():
                path = folder / f"{tid}.{ext}"
                row = dict(split=split, trip=tid, file=f"{tid}.{ext}", exists=path.exists(),
                           size_kb=round(path.stat().st_size / 1024, 1) if path.exists() else None)
                if path.exists():
                    df = read_raw(path, cols)
                    frames[ext] = df
                    row.update(rows=len(df), columns=df.shape[1],
                               dtypes=";".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items()),
                               missing_cells=int(df.isna().sum().sum()))
                file_rows.append(row)

            track = frames.get("track")
            nodes = frames.get("nodes")
            arcs = frames.get("arcs")
            route = frames.get("route")
            q = dict(split=split, trip=tid)
            if track is not None:
                dt = track["time_s"].diff().dropna()
                moved = np.hypot(track["lon"].diff(), track["lat"].diff()).dropna()
                q.update(
                    gps_points=len(track),
                    gps_missing_cells=int(track.isna().sum().sum()),
                    gps_invalid_lon_lat=int(((track["lon"].abs() > 180) | (track["lat"].abs() > 90)).sum()),
                    duplicate_timestamps=int(track["time_s"].duplicated().sum()),
                    non_increasing_steps=int((dt <= 0).sum()),
                    stationary_consecutive_points=int((moved == 0).sum()),
                    steps_exactly_1s=float((dt == 1).mean()) if len(dt) else np.nan,
                    gaps_over_5s=int((dt > 5).sum()),
                    gaps_over_60s=int((dt > 60).sum()),
                )
            if nodes is not None:
                q["node_missing_cells"] = int(nodes.isna().sum().sum())
            if arcs is not None and nodes is not None:
                q["arcs_bad_node_ids"] = int(((arcs < 0) | (arcs >= len(nodes))).any(axis=1).sum())
                q["self_loop_arcs"] = int((arcs["from_node"] == arcs["to_node"]).sum())
                q["duplicate_arcs"] = int(arcs.duplicated().sum())
            if route is not None and arcs is not None:
                r = route["arc_id"].to_numpy()
                valid = (r >= 0) & (r < len(arcs))
                q["route_arcs"] = len(r)
                q["route_invalid_arc_ids"] = int((~valid).sum())
                q["route_repeated_arcs"] = int(len(r) - len(np.unique(r)))
                if valid.all() and len(r) > 1:
                    a = arcs.to_numpy()
                    # consecutive route arcs are connected if one ends where the next starts
                    q["route_consecutive_connected"] = float(np.mean(a[r[:-1], 1] == a[r[1:], 0]))
            quality_rows.append(q)
    return pd.DataFrame(file_rows), pd.DataFrame(quality_rows)


def partition_summary(inventory):
    cols = ["gps_points", "median_step_s", "max_gap_s", "gaps_over_5s", "gps_path_m",
            "nodes", "arcs", "weak_components", "isolated_nodes", "route_arcs",
            "alignment_within_50m", "candidate_miss_rate"]
    summary = inventory.groupby("split")[cols].describe().T
    counts = inventory.groupby("split").size().rename("trips")
    return summary, counts


# ---------------------------------------------------------------- plots
def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


def plot_partition_overview(inventory, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split, color in zip(SPLITS, ["tab:blue", "tab:red"]):
        part = inventory[inventory.split == split]
        axes[0].hist(part.gps_points, bins=25, alpha=0.6, color=color, label=split)
        axes[1].hist(np.log10(part.arcs), bins=25, alpha=0.6, color=color, label=split)
        axes[2].hist(part.weak_components, bins=25, alpha=0.6, color=color, label=split)
    axes[0].set(title="GPS points per trip", xlabel="points", ylabel="trips")
    axes[1].set(title="Road segments per trip", xlabel="log10(arcs)")
    axes[2].set(title="Road-network connectivity", xlabel="weakly connected components")
    for ax in axes:
        ax.legend()
    save(fig, out, "partition_overview.png")


def plot_time_gaps(root, out):
    """Sampling-interval distribution, training partition only."""
    gaps = []
    for folder in trip_folders(root, "train"):
        t = read_raw(folder / f"{folder.name}.track", FILES["track"])["time_s"].to_numpy()
        gaps.append(np.diff(t))
    gaps = np.concatenate(gaps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    values, counts = np.unique(np.clip(gaps, 0, 10), return_counts=True)
    axes[0].bar(values, counts / counts.sum())
    axes[0].set(title="Time step between GPS points (train, clipped at 10 s)",
                xlabel="seconds", ylabel="fraction of steps")
    big = gaps[gaps > 5]
    if len(big):
        axes[1].hist(np.log10(big), bins=30)
    axes[1].set(title=f"Gaps > 5 s (train): {len(big)} gaps", xlabel="log10(gap seconds)", ylabel="count")
    save(fig, out, "time_gaps_train.png")
    return dict(total_steps=int(len(gaps)), fraction_exactly_1s=float(np.mean(gaps == 1)),
                gaps_over_5s=int(len(big)), max_gap_s=float(gaps.max()))


def plot_missing(quality, out):
    cols = ["gps_missing_cells", "gps_invalid_lon_lat", "duplicate_timestamps",
            "non_increasing_steps", "stationary_consecutive_points", "gaps_over_5s",
            "route_invalid_arc_ids", "duplicate_arcs", "self_loop_arcs"]
    cols = [c for c in cols if c in quality]
    totals = quality.groupby("split")[cols].sum().T
    fig, ax = plt.subplots(figsize=(10, 4.5))
    totals.plot.barh(ax=ax)
    ax.set(title="Missing / invalid / anomalous values (summed over trips)", xlabel="count")
    save(fig, out, "missing_and_invalid.png")
    return totals


def plot_trajectories(root, out, n=6):
    """GPS trajectories in metres for a few training trips (each in its own local frame)."""
    folders = trip_folders(root, "train")[:n]
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    for ax, folder in zip(axes.ravel(), folders):
        tr = load_trip(folder)
        ax.plot(tr.gps[:, 0], tr.gps[:, 1], ".", ms=1.5)
        ax.plot(*tr.gps[0], "go", label="start")
        ax.plot(*tr.gps[-1], "rs", label="end")
        ax.set(title=f"train/{tr.id} ({len(tr.gps)} pts)", xlabel="east (m)", ylabel="north (m)")
        ax.set_aspect("equal")
    axes.ravel()[0].legend()
    save(fig, out, "gps_trajectories_train.png")


def nearby_arcs(tr, margin_m=300):
    lo = tr.gps.min(axis=0) - margin_m
    hi = tr.gps.max(axis=0) + margin_m
    a, b = tr.xy[tr.arcs[:, 0]], tr.xy[tr.arcs[:, 1]]
    inside = np.all((a >= lo) & (a <= hi), axis=1) | np.all((b >= lo) & (b <= hi), axis=1)
    return np.flatnonzero(inside)


def plot_road_network(folder, out):
    tr = load_trip(folder)
    route = read_raw(folder / f"{tr.id}.route", FILES["route"])["arc_id"].to_numpy()
    keep = nearby_arcs(tr)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in keep:
        s, e = tr.xy[tr.arcs[i]]
        ax.plot([s[0], e[0]], [s[1], e[1]], color="0.8", lw=0.6)
    for i in route:
        s, e = tr.xy[tr.arcs[i]]
        ax.plot([s[0], e[0]], [s[1], e[1]], color="orange", lw=3)
    ax.plot(tr.gps[:, 0], tr.gps[:, 1], ".", ms=2, color="tab:blue")
    ax.plot([], [], color="0.8", label=f"road network ({len(keep)} arcs near track)")
    ax.plot([], [], color="orange", lw=3, label=f"reference route ({len(route)} arcs)")
    ax.plot([], [], ".", color="tab:blue", label=f"raw GPS ({len(tr.gps)} points)")
    ax.set(title=f"{folder.parent.name}/{tr.id}: road network, GPS and reference",
           xlabel="east (m)", ylabel="north (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    save(fig, out, f"road_network_{folder.parent.name}_{tr.id}.png")


def folium_map(folder, out):
    """Raw-data map: road network near the track, raw GPS, reference route. No predictions."""
    tr = load_trip(folder)
    route = read_raw(folder / f"{tr.id}.route", FILES["route"])["arc_id"].to_numpy()
    latlon = lambda node: [float(tr.nodes[node, 1]), float(tr.nodes[node, 0])]
    center = [float(tr.track[:, 1].mean()), float(tr.track[:, 0].mean())]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    roads = folium.FeatureGroup(name="Road network (near track)")
    lines = [[latlon(s), latlon(e)] for s, e in tr.arcs[nearby_arcs(tr)]]
    if lines:
        folium.PolyLine(lines, color="gray", weight=1.5, opacity=0.6).add_to(roads)
    roads.add_to(m)

    ref = folium.FeatureGroup(name="Reference route")
    folium.PolyLine([[latlon(tr.arcs[i, 0]), latlon(tr.arcs[i, 1])] for i in route],
                    color="green", weight=5, opacity=0.8).add_to(ref)
    ref.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS observations")
    for lon, lat, t in tr.track:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True,
                            fill_opacity=0.8, tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)

    folium.Marker([tr.track[0, 1], tr.track[0, 0]], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([tr.track[-1, 1], tr.track[-1, 0]], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
              'padding:8px;border:1px solid #999;font-size:13px">'
              f'<b>{folder.parent.name}/{tr.id}</b><br>'
              '<span style="color:blue">&#9679;</span> Raw GPS<br>'
              '<span style="color:green">&#9644;</span> Reference route<br>'
              '<span style="color:gray">&#9644;</span> Road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr.track[:, 1].min(), tr.track[:, 0].min()],
                  [tr.track[:, 1].max(), tr.track[:, 0].max()]])
    path = out / f"eda_map_{folder.parent.name}_{tr.id}.html"
    m.save(str(path))
    return path


# ---------------------------------------------------------------- main
def run(data_root, out, trip, split):
    root, out = Path(data_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # prefer the inventory from your own prepare run, else the committed copy
    inventory_path = next((p for p in [Path("work/cache/inventory.csv"), Path("results/inventory.csv")]
                           if p.exists()), None)
    if inventory_path is None:
        raise SystemExit("inventory.csv not found: run member1_data.prepare first")
    print(f"Using {inventory_path}")
    inventory = pd.read_csv(inventory_path, dtype={"trip": str})

    files, quality = file_inventory_and_quality(root)
    files.to_csv(out / "file_inventory.csv", index=False)
    quality.to_csv(out / "data_quality_per_trip.csv", index=False)
    summary, counts = partition_summary(inventory)
    summary.to_csv(out / "summary_by_partition.csv")

    plot_partition_overview(inventory, out)
    gap_stats = plot_time_gaps(root, out)
    missing_totals = plot_missing(quality, out)
    plot_trajectories(root, out)

    folder = root / split / trip
    if not folder.is_dir():
        raise SystemExit(f"Trip folder not found: {folder}")
    plot_road_network(folder, out)
    map_path = folium_map(folder, out)

    anomalies = inventory.sort_values("alignment_within_50m").head(5)[
        ["split", "trip", "gps_points", "max_gap_s", "alignment_within_50m", "candidate_miss_rate"]]
    report = dict(
        partitions={k: int(v) for k, v in counts.items()},
        evaluator_testing_used=False,
        train_sampling=gap_stats,
        missing_and_invalid_totals=missing_totals.to_dict(),
        route_consecutive_connected_mean=float(quality["route_consecutive_connected"].mean())
        if "route_consecutive_connected" in quality else None,
        worst_alignment_trips=anomalies.to_dict(orient="records"),
        folium_map=str(map_path),
    )
    (out / "eda_summary.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"Trips per partition: {report['partitions']}")
    print(f"Train time steps exactly 1 s: {gap_stats['fraction_exactly_1s']:.3f}, "
          f"gaps > 5 s: {gap_stats['gaps_over_5s']}, max gap: {gap_stats['max_gap_s']:.0f} s")
    print("Missing / invalid totals:\n" + missing_totals.to_string())
    print(f"Consecutive reference arcs connected (mean): {report['route_consecutive_connected_mean']}")
    print("Worst-aligned trips:\n" + anomalies.to_string(index=False))
    print(f"Outputs written to {out}/ (map: {map_path.name})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="results/eda")
    p.add_argument("--trip", default="00000039", help="trip used for the road-network plot and Folium map")
    p.add_argument("--split", default="train", choices=SPLITS)
    a = p.parse_args()
    run(a.data_root, a.out, a.trip, a.split)
