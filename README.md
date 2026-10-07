# Spatial analysis algorithms for multiplexed immunofluorescence (MACSima) data

Python implementation of the spatial analysis algorithms used in

> Manuscript in preparation.

The functions take cell coordinates (in µm) with cell-type labels or boolean masks and return per-cell assignments
(niches, neighborhoods, zones, distances to the tumor edge) or per-sample values; the statistics module compares
per-sample values between groups or conditions.

## Installation

Python 3.12 (tested with 3.12.3 on Windows 11; no special hardware required).

```bash
git clone https://github.com/<user>/<repository>.git
cd <repository>
python -m pip install -r requirements.txt
```

Installation takes a few minutes.

## Modules

| Module | Functions | Method |
|---|---|---|
| `tiling` | `tile_counts`, `tiling_metrics`, `variance_to_mean_ratio`, `gini`, `morisita_index`, `occupancy`, `tissue_area_mm2`, `density_per_mm2`, `convex_hull_area_mm2` | A square grid (default 50 µm) is laid over the sample and tiles containing at least one cell are kept. Per cell type: variance-to-mean ratio of the tile counts (patchiness), Gini index, Morisita index of dispersion and fraction of occupied tiles; densities in cells per mm² of occupied tiles. |
| `neighborhood` | `knn_graph`, `neighborhood_enrichment`, `pair_enrichment`, `mean_nearest_distance` | Symmetric k-nearest-neighbor graph (default k = 15). For each pair of cell types, the number of neighboring cells is compared with random permutations of the labels (default 500) and expressed as a z-score, as in squidpy `nhood_enrichment` (Palla et al., *Nat. Methods* 2022). Mean distance from each cell of one population to the nearest cell of another. |
| `niches` | `local_fractions`, `assign_niches`, `fraction_by_niche` | Each cell is assigned to a cancer, macrophage, dendritic-cell or fibroblast niche from the composition of its k nearest neighbors divided by the frequency of each lineage in the sample, using lineage-specific enrichment thresholds. |
| `spiat` | `identify_neighborhoods`, `neighborhood_summary`, `enrichment_zones`, `mean_nearest_neighbor_distance` | Neighborhoods follow SPIAT (Feng et al., *Nat. Commun.* 2023): cells of interest are clustered with DBSCAN using a radius of three times the mean nearest-neighbor distance of all cells and a minimum of 10 cells per neighborhood (one label per cell of interest; -1 = not in a neighborhood). Enrichment zones: each cell receives the lineage with the highest local fraction within the same radius relative to its frequency in the sample. |
| `tumor_edge` | `otsu_threshold`, `signed_distance`, `density_profile`, `peak_distance` | Cancer-cell counts on the grid are smoothed with a Gaussian kernel (default σ = 40 µm) and thresholded (Otsu) to define the tumor area. Each cell receives the signed distance of its tile to the tumor edge (negative inside). Cell densities are computed in distance bins (default 10 µm, -200 to +300 µm) and the peak is the maximum of a three-bin moving average. |
| `statistics` | `pairwise_tukey`, `pairwise_mann_whitney`, `kruskal_wallis`, `mann_whitney`, `paired_t`, `pairwise_paired_t`, `repeated_measures_anova`, `mean`, `sd`, `sem` | Tukey's HSD between groups (pairwise Mann-Whitney U when a group has fewer than two values; `pairwise_tukey` returns the pairs and the test used), two-sided Mann-Whitney U with Cliff's δ, Kruskal-Wallis with ε², two-sided paired t-test and one-way repeated-measures ANOVA on per-sample values. |

## Demo

`example/mc38_apd1_cells.csv` contains the cell positions (µm) and cell types of one MC38 tumor (αPD1 group).
`demo.py` runs the algorithms on it:

```bash
python demo.py
```

Expected output (run time about 1 s on a desktop computer):

```
Cells: 17814
Tiling cancer     VMR 8.34  Gini 0.56  Morisita 1.91  occupancy 0.79
Tiling macrophage VMR 3.21  Gini 0.71  Morisita 2.63  occupancy 0.46
Tiling CD8        VMR 1.55  Gini 0.88  Morisita 3.71  occupancy 0.15
Neighborhood enrichment z: cancer-cancer 187.9, cancer-macrophage -70.1, CD8-macrophage 3.1
CD8 to nearest cancer cell: 17.6 um
Niches: {'cancer': 7078, 'macrophage': 4519, 'dendritic': 3789, 'fibroblast': 2428}
CD8 T cells per niche (%): {'cancer': 23.7, 'macrophage': 33.1, 'dendritic': 25.4, 'fibroblast': 17.8}
SPIAT neighborhoods: radius 19.7 um, 13 neighborhoods, 170 of 2119 immune cells
Density peak relative to the tumor edge (um): {'cancer': -175.0, 'macrophage': 235.0, 'dendritic': -175.0, 'CD8': -165.0}
```

The niche thresholds in `demo.py` are example values.

## Use on other data

Pass cell coordinates in µm (`xy`, an n × 2 array) with cell-type labels or one boolean mask per population, as in
`demo.py`. Parameter values used in the article are given in its Methods.

## Citation

Please cite the article above when using this code.
