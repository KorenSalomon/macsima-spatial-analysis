"""Fishnet tiling: per-tile cell counts, patchiness (VMR), Gini index, Morisita index, tile occupancy and densities."""
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def make_grid(xy, tile_um=50.0):
    xy = np.asarray(xy, dtype=float)
    if len(xy) == 0:
        raise ValueError("xy is empty")
    x0, x1 = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
    y0, y1 = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
    nx = max(1, int(math.ceil((x1 - x0) / tile_um)))
    ny = max(1, int(math.ceil((y1 - y0) / tile_um)))
    return {"x0": x0, "y0": y0, "nx": nx, "ny": ny, "tile_um": float(tile_um), "shape": (ny, nx)}


def tile_index(xy, grid):
    xy = np.asarray(xy, dtype=float)
    t = grid["tile_um"]
    ix = np.clip(((xy[:, 0] - grid["x0"]) / t).astype(int), 0, grid["nx"] - 1)
    iy = np.clip(((xy[:, 1] - grid["y0"]) / t).astype(int), 0, grid["ny"] - 1)
    return ix, iy


def tile_counts(xy, mask, tile_um=50.0):
    grid = make_grid(xy, tile_um)
    ix, iy = tile_index(xy, grid)
    nx, ny = grid["nx"], grid["ny"]
    flat = iy * nx + ix
    occ = np.zeros(nx * ny)
    np.add.at(occ, flat, 1)
    in_tissue = occ > 0
    m = np.asarray(mask, bool)
    cc = np.zeros(nx * ny)
    if m.any():
        np.add.at(cc, flat[m], 1)
    return cc[in_tissue]


def variance_to_mean_ratio(counts):
    x = np.asarray(counts, dtype=float)
    mean = x.mean() if x.size else np.nan
    return float(x.var() / mean) if (x.size and mean > 0) else np.nan


def gini(counts):
    x = np.asarray(counts, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * x.sum()))


def morisita_index(counts):
    x = np.asarray(counts, dtype=float)
    n_tiles = x.size
    total = x.sum()
    return float(n_tiles * np.sum(x * (x - 1)) / (total * (total - 1))) if total > 1 else np.nan


def occupancy(counts):
    x = np.asarray(counts, dtype=float)
    return float(np.mean(x > 0)) if x.size else np.nan


def tiling_metrics(xy, mask, tile_um=50.0):
    cc = tile_counts(xy, mask, tile_um)
    return {"n_cells": int(np.asarray(mask, bool).sum()), "VMR": variance_to_mean_ratio(cc),
            "Gini": gini(cc) if cc.size else np.nan, "Morisita": morisita_index(cc), "occupancy": occupancy(cc)}


def tissue_area_mm2(xy, tile_um=50.0):
    cc = tile_counts(xy, np.ones(len(xy), bool), tile_um)
    return cc.size * (tile_um ** 2) / 1e6


def density_per_mm2(xy, mask, tile_um=50.0):
    area = tissue_area_mm2(xy, tile_um)
    return int(np.asarray(mask, bool).sum()) / area if area > 0 else np.nan


def convex_hull_area_mm2(xy):
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return np.nan
    try:
        return float(ConvexHull(xy).volume) / 1e6
    except QhullError:
        return np.nan
