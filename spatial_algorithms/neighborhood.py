"""Neighborhood enrichment on a k-nearest-neighbor graph (label permutation z-scores) and nearest-neighbor distances."""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def knn_graph(xy, k=15):
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    if n <= k:
        raise ValueError("knn_graph needs more than k cells")
    _, nn = cKDTree(xy).query(xy, k=k + 1)
    nn = nn[:, 1:]
    rows = np.repeat(np.arange(n), nn.shape[1])
    cols = nn.ravel()
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    return A


def neighborhood_enrichment(xy, cell_type, types, k=15, n_perm=500, seed=0):
    cell_type = np.asarray(cell_type, dtype=object)
    n = len(cell_type)
    P = len(types)
    index = {t: i for i, t in enumerate(types)}
    lab = np.array([index.get(t, P) for t in cell_type], dtype=int)
    counts = np.array([int((cell_type == t).sum()) for t in types])
    Z = np.full((P, P), np.nan)
    if n < k + 2:
        return Z, counts
    A = knn_graph(xy, k)

    def cooccurrence(lv):
        L = sparse.csr_matrix((np.ones(n), (np.arange(n), lv)), shape=(n, P + 1))[:, :P]
        return np.asarray((L.T @ (A @ L)).todense())

    obs = cooccurrence(lab)
    rng = np.random.default_rng(seed)
    acc = np.zeros((P, P))
    acc2 = np.zeros((P, P))
    for _ in range(n_perm):
        rng.shuffle(lab)
        c = cooccurrence(lab)
        acc += c
        acc2 += c * c
    mean = acc / n_perm
    std = np.sqrt(np.maximum(acc2 / n_perm - mean * mean, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(std > 0, (obs - mean) / std, np.nan)
    return Z, counts


def pair_enrichment(xy, mask_a, mask_b, k=15, n_perm=500, seed=0):
    mask_a = np.asarray(mask_a, bool)
    mask_b = np.asarray(mask_b, bool)
    n = len(mask_a)
    na, nb = int(mask_a.sum()), int(mask_b.sum())
    if n < k + 2 or na == 0 or nb == 0:
        return np.nan, na, nb
    lab = np.full(n, 2, dtype=int)
    lab[mask_a] = 0
    lab[mask_b] = 1
    A = knn_graph(xy, k)

    def ab(lv):
        L = sparse.csr_matrix((np.ones(n), (np.arange(n), lv)), shape=(n, 3))[:, :2]
        return float(np.asarray((L.T @ (A @ L)).todense())[0, 1])

    obs = ab(lab)
    rng = np.random.default_rng(seed)
    acc = acc2 = 0.0
    for _ in range(n_perm):
        rng.shuffle(lab)
        c = ab(lab)
        acc += c
        acc2 += c * c
    mean = acc / n_perm
    std = max(acc2 / n_perm - mean * mean, 0.0) ** 0.5
    return ((obs - mean) / std if std > 0 else np.nan), na, nb


def mean_nearest_distance(xy, mask_from, mask_to):
    xy = np.asarray(xy, dtype=float)
    a = xy[np.asarray(mask_from, bool)]
    b = xy[np.asarray(mask_to, bool)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, len(a), len(b)
    d, _ = cKDTree(b).query(a, k=1)
    return float(np.mean(d)), int(len(a)), int(len(b))
