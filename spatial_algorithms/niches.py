"""Cellular niches: each cell is assigned to a cancer, macrophage, dendritic-cell or fibroblast niche from the
composition of its k nearest neighbors relative to the sample average."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

LINEAGES = ["cancer", "macrophage", "dendritic", "fibroblast"]
MINORITY = ["dendritic", "macrophage", "fibroblast"]


def local_fractions(xy, lineage_masks, k=15):
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    k_eff = int(min(k, max(1, n - 1)))
    _, idx = NearestNeighbors(n_neighbors=min(k_eff + 1, n)).fit(xy).kneighbors(xy)
    idx = idx[:, 1:]
    return {lab: (np.asarray(lineage_masks[lab]).astype(float)[idx].mean(axis=1) if idx.shape[1] else np.zeros(n))
            for lab in LINEAGES}


def assign_niches(xy, lineage_masks, *, thresholds, cancer_threshold, cancer_tie, k=15):
    sens = {lab: float(thresholds[lab]) for lab in MINORITY}
    n = len(xy)
    fr = local_fractions(xy, lineage_masks, k)
    freq = {lab: float(np.asarray(lineage_masks[lab]).mean()) for lab in LINEAGES}
    e = {lab: (fr[lab] / freq[lab] if freq[lab] > 0 else np.zeros(n)) for lab in LINEAGES}
    e_cancer = e["cancer"]
    e_dc_mac = np.column_stack([e["dendritic"], e["macrophage"]])
    emat = np.column_stack([e[lab] for lab in MINORITY])
    sv = np.array([float(sens[lab]) for lab in MINORITY])
    passing = emat >= sv[None, :]
    any_pass = passing.any(axis=1)
    best = np.argmax(np.where(passing, emat, -np.inf), axis=1)
    niche = np.array(["cancer"] * n, dtype=object)
    for j, lab in enumerate(MINORITY):
        niche[any_pass & (best == j)] = lab
    best_val = np.where(any_pass, emat[np.arange(n), best], -np.inf)
    is_dc_mac = np.isin(best, [0, 1]) & any_pass
    niche[is_dc_mac & (e_cancer >= cancer_threshold) & (e_cancer >= cancer_tie * best_val)] = "cancer"
    lead = np.argmax(e_dc_mac, axis=1)
    lead_val = e_dc_mac[np.arange(n), lead]
    lead_lab = np.where(lead == 0, "dendritic", "macrophage")
    to_minor = ~any_pass & (e_cancer < cancer_threshold) & (lead_val > 1.0)
    niche[to_minor] = lead_lab[to_minor]
    return niche


def fraction_by_niche(population_mask, niche, niches=LINEAGES):
    m = np.asarray(population_mask, bool)
    niche = np.asarray(niche, dtype=object)
    counts = {z: int(np.sum(m & (niche == z))) for z in niches}
    total = sum(counts.values())
    return {z: (100.0 * counts[z] / total if total > 0 else np.nan) for z in niches}
