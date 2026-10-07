"""Group and within-sample statistics on per-sample values."""
import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import kruskal, mannwhitneyu, ttest_rel, tukey_hsd


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def mean(a):
    a = _finite(a)
    return float(np.mean(a)) if len(a) else np.nan


def sd(a):
    a = _finite(a)
    return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def sem(a):
    a = _finite(a)
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def kruskal_wallis(values_by_group):
    arrs = [_finite(v) for v in values_by_group.values()]
    arrs = [a for a in arrs if len(a) >= 1]
    N = sum(len(a) for a in arrs)
    if len(arrs) < 2 or all(len(a) < 2 for a in arrs) or N < 3:
        return np.nan, np.nan
    try:
        res = kruskal(*arrs)
    except ValueError:
        return np.nan, np.nan
    return float(res.pvalue), float(res.statistic) / (N - 1)


def mann_whitney(control, other):
    a, b = _finite(control), _finite(other)
    if len(a) < 1 or len(b) < 1:
        return np.nan, np.nan
    gt = sum((y > x) for y in b for x in a)
    lt = sum((y < x) for y in b for x in a)
    cliffs_delta = (gt - lt) / (len(a) * len(b))
    try:
        p = float(mannwhitneyu(b, a, alternative="two-sided").pvalue)
    except ValueError:
        p = np.nan
    return p, float(cliffs_delta)


def pairwise_tukey(values_by_group, groups):
    arrs = [_finite(values_by_group.get(g, [])) for g in groups]
    if all(len(a) >= 2 for a in arrs):
        try:
            res = tukey_hsd(*arrs)
            return [(groups[i], groups[j], float(res.pvalue[i, j]))
                    for i in range(len(groups)) for j in range(i + 1, len(groups))], "Tukey HSD"
        except ValueError:
            pass
    return [(groups[i], groups[j], mann_whitney(arrs[i], arrs[j])[0])
            for i in range(len(groups)) for j in range(i + 1, len(groups))], "Mann-Whitney"


def pairwise_mann_whitney(values_by_group, groups):
    arrs = [_finite(values_by_group.get(g, [])) for g in groups]
    out = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = arrs[i], arrs[j]
            try:
                p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue) if (len(a) and len(b)) else np.nan
            except ValueError:
                p = np.nan
            out.append((groups[i], groups[j], p))
    return out


def paired_t(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 2:
        return np.nan
    try:
        return float(ttest_rel(a[m], b[m]).pvalue)
    except ValueError:
        return np.nan


def pairwise_paired_t(values_by_condition, conditions):
    return [(conditions[i], conditions[j],
             paired_t(values_by_condition.get(conditions[i], []), values_by_condition.get(conditions[j], [])))
            for i in range(len(conditions)) for j in range(i + 1, len(conditions))]


def repeated_measures_anova(values_by_condition, conditions):
    rows = [list(values_by_condition.get(c, [])) for c in conditions]
    if len(rows) < 2 or len({len(r) for r in rows}) != 1:
        raise ValueError("every condition needs the same number of samples")
    M = np.array(rows, float)
    Y = M.T
    Y = Y[~np.any(~np.isfinite(Y), axis=1)]
    n, k = Y.shape
    if n < 2 or k < 2:
        return np.nan
    grand = Y.mean()
    ss_cond = n * np.sum((Y.mean(0) - grand) ** 2)
    ss_subj = k * np.sum((Y.mean(1) - grand) ** 2)
    ss_err = np.sum((Y - grand) ** 2) - ss_cond - ss_subj
    df_cond, df_err = k - 1, (k - 1) * (n - 1)
    if df_err <= 0 or ss_err <= 0:
        return np.nan
    F = (ss_cond / df_cond) / (ss_err / df_err)
    return float(f_dist.sf(F, df_cond, df_err)) if np.isfinite(F) else np.nan
