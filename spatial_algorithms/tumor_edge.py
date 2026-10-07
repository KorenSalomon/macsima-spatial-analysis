"""Tumor mask from the smoothed cancer-cell density, signed distance of each cell to the tumor edge, cell-density
profiles along that distance and the distance of each profile's peak."""
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter

from .tiling import make_grid, tile_index


def otsu_threshold(values):
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0 or float(np.max(x)) == float(np.min(x)):
        return (float(np.max(x)) if x.size else 0.0), True
    hist, edges = np.histogram(x, bins=256)
    p = hist.astype(float) / max(hist.sum(), 1)
    cum = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * mids)
    denom = cum * (1.0 - cum)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu[-1] * cum - mu) ** 2 / denom
    return float(mids[int(np.nanargmax(sigma_b2))]), False


def signed_distance(xy, cancer_mask, tile_um=50.0, bandwidth_um=40.0):
    xy = np.asarray(xy, dtype=float)
    grid = make_grid(xy, tile_um)
    ny, nx = grid["shape"]
    tumor_pts = xy[np.asarray(cancer_mask, bool)]
    if len(tumor_pts) < 10:
        return None
    H = np.zeros((ny, nx))
    tix, tiy = tile_index(tumor_pts, grid)
    np.add.at(H, (tiy, tix), 1)
    dens = gaussian_filter(H, sigma=max(bandwidth_um / tile_um, 0.6))
    thr, degenerate = otsu_threshold(dens[dens > 0]) if np.any(dens > 0) else (0, True)
    mask = dens > thr
    frac = mask.mean()
    if degenerate or frac < 0.01 or frac > 0.95:
        thr = np.percentile(dens[dens > 0], 70) if np.any(dens > 0) else 0
        mask = dens > thr
        frac = mask.mean()
    if frac <= 0 or frac >= 1:
        return None
    mask = binary_fill_holes(mask)
    field = (distance_transform_edt(~mask) - distance_transform_edt(mask)) * tile_um
    cix, ciy = tile_index(xy, grid)
    occupied = np.zeros(ny * nx)
    np.add.at(occupied, ciy * nx + cix, 1)
    return {"per_cell": field[ciy, cix], "field": field, "tumor_mask": mask,
            "occupied": occupied.reshape(ny, nx) > 0, "tile_area_mm2": tile_um ** 2 / 1e6}


def density_profile(sd, population_masks, bins=None):
    if bins is None:
        bins = np.arange(-200, 305, 10.0)
    d = sd["per_cell"]
    field = sd["field"]
    occupied = sd["occupied"]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        area = np.sum(((field >= lo) & (field < hi)) & occupied) * sd["tile_area_mm2"]
        in_bin = (d >= lo) & (d < hi)
        row = {"distance_um": 0.5 * (lo + hi)}
        for name, m in population_masks.items():
            row[name] = (np.sum(np.asarray(m, bool) & in_bin) / area) if area > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def peak_distance(profile, column, window=3):
    s = profile.sort_values("distance_um").set_index("distance_um")[column]
    s = s.rolling(window, center=True, min_periods=1).mean()
    return float(s.idxmax()) if s.notna().any() else np.nan
