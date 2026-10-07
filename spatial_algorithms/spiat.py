"""Neighborhoods following SPIAT (Feng et al., Nat. Commun. 2023): proximity clustering of cells of interest and
radius-based local enrichment zones."""
from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def mean_nearest_neighbor_distance(xy):
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return np.nan
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(d[:, 1].mean())


def identify_neighborhoods(xy, mask_of_interest, radius_factor=3.0, min_size=10):
    xy = np.asarray(xy, dtype=float)
    mask = np.asarray(mask_of_interest, bool)
    radius = radius_factor * mean_nearest_neighbor_distance(xy)
    pts = xy[mask]
    labels = np.full(len(pts), -1, dtype=int)
    if len(pts) >= min_size and np.isfinite(radius) and radius > 0:
        labels = DBSCAN(eps=radius, min_samples=min_size).fit(pts).labels_.astype(int)
        for cl in set(labels) - {-1}:
            if int((labels == cl).sum()) < min_size:
                labels[labels == cl] = -1
        relabel = {c: i for i, c in enumerate(sorted(set(labels) - {-1}))}
        labels = np.array([relabel.get(int(v), -1) for v in labels], dtype=int)
    return labels, radius


def neighborhood_summary(labels):
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    n_nb = len(set(labels.tolist()) - {-1})
    inside = int((labels >= 0).sum())
    return {"n_cells_of_interest": n, "n_neighborhoods": n_nb, "n_cells_in_neighborhoods": inside,
            "pct_in_neighborhoods": (100.0 * inside / n) if n else np.nan,
            "mean_neighborhood_size": (inside / n_nb) if n_nb else np.nan}


def enrichment_zones(xy, lineage_masks, radius_factor=3.0):
    xy = np.asarray(xy, dtype=float)
    masks = OrderedDict(lineage_masks)
    names = list(masks)
    n = len(xy)
    tree = cKDTree(xy)
    radius = radius_factor * mean_nearest_neighbor_distance(xy)
    total = tree.query_ball_point(xy, radius, return_length=True).astype(float)
    total[total <= 0] = 1.0
    E = np.full((n, len(names)), -1.0)
    for j, name in enumerate(names):
        m = np.asarray(masks[name], bool)
        freq = float(m.mean())
        if freq <= 0 or int(m.sum()) == 0:
            continue
        local = cKDTree(xy[m]).query_ball_point(xy, radius, return_length=True).astype(float)
        E[:, j] = (local / total) / freq
    return np.array([names[j] for j in np.argmax(E, axis=1)], dtype=object), radius
