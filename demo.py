"""Demo of the spatial algorithms on one MC38 tumor (example/mc38_apd1_cells.csv)."""
import os
import time

import numpy as np
import pandas as pd

from spatial_algorithms import neighborhood, niches, spiat, tiling, tumor_edge

start = time.time()
cells = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "example", "mc38_apd1_cells.csv"))
xy = cells[["x_um", "y_um"]].to_numpy(float)
cell_type = cells["cell_type"].to_numpy(object)
masks = {t: cell_type == t for t in ["cancer", "macrophage", "dendritic", "fibroblast", "CD8", "CD4"]}
print("Cells: %d" % len(cells))

for t in ["cancer", "macrophage", "CD8"]:
    m = tiling.tiling_metrics(xy, masks[t])
    print("Tiling %-10s VMR %.2f  Gini %.2f  Morisita %.2f  occupancy %.2f"
          % (t, m["VMR"], m["Gini"], m["Morisita"], m["occupancy"]))

types = ["cancer", "macrophage", "dendritic", "fibroblast", "CD8", "CD4"]
z, counts = neighborhood.neighborhood_enrichment(xy, cell_type, types, n_perm=100)
print("Neighborhood enrichment z: cancer-cancer %.1f, cancer-macrophage %.1f, CD8-macrophage %.1f"
      % (z[0, 0], z[0, 1], z[4, 1]))
print("CD8 to nearest cancer cell: %.1f um" % neighborhood.mean_nearest_distance(xy, masks["CD8"], masks["cancer"])[0])

niche = niches.assign_niches(xy, {t: masks[t] for t in niches.LINEAGES},
                             thresholds={"dendritic": 2.0, "macrophage": 2.0, "fibroblast": 2.0},
                             cancer_threshold=1.5, cancer_tie=1.2)
print("Niches:", {k: int((niche == k).sum()) for k in niches.LINEAGES})
print("CD8 T cells per niche (%):", {k: round(v, 1) for k, v in niches.fraction_by_niche(masks["CD8"], niche).items()})

immune = np.isin(cell_type, ["macrophage", "dendritic", "CD8", "CD4", "Treg"])
labels, radius = spiat.identify_neighborhoods(xy, immune)
s = spiat.neighborhood_summary(labels)
print("SPIAT neighborhoods: radius %.1f um, %d neighborhoods, %d of %d immune cells"
      % (radius, s["n_neighborhoods"], s["n_cells_in_neighborhoods"], s["n_cells_of_interest"]))

sd = tumor_edge.signed_distance(xy, masks["cancer"])
profile = tumor_edge.density_profile(sd, {t: masks[t] for t in ["cancer", "macrophage", "dendritic", "CD8"]})
print("Density peak relative to the tumor edge (um):",
      {t: tumor_edge.peak_distance(profile, t) for t in ["cancer", "macrophage", "dendritic", "CD8"]})

print("Run time: %.1f s" % (time.time() - start))
