# Forward-Matching Mesh-Separation Validation

The template library is the existing FDM cache generated on the standard
`dx_factor=0.25` mesh. The separated test generates synthetic observations
on a finer `dx_factor=0.125` mesh before matching them to the unchanged
template cache.

| Observation mesh | MAE (deg) | Median AE (deg) | <=5 deg (%) | <=10 deg (%) | Family acc. (%) | Mean corr. |
|---|---:|---:|---:|---:|---:|---:|
| same_mesh_dx0.25 | 4.20 | 3.00 | 80.0 | 100.0 | 60.0 | 0.979 |
| separated_fine_mesh_dx0.125 | 4.40 | 3.00 | 76.0 | 100.0 | 60.0 | 0.979 |
