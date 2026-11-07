# Release validation

- The unchanged German component passed its release validator (112 manifest files, six unlabelled input arrays, 32 benchmark conditions and 32 GPS conditions). Its aggregate report regenerated successfully.
- The Spanish label attachment was tested against the authorized working archive. The exact SOC array hash and every retained predictor field matched. A non-cohort below-detection row was ignored.
- The archived Spanish splits passed disjointness and membership checks. There were 40 outer splits in each regime, with identical test membership and reduced training subsets drawn from their corresponding full training sets.
- The Spanish block-bootstrap contrasts, NDVI slopes and quartiles, training-size effects and high-SOC error shares were regenerated from the archived held-out predictions. The aggregate CSVs matched within numeric tolerance.
- The main Tables 2–3 and Supplementary Tables S10–S14 were regenerated from archived aggregates. One endpoint copied into the supplied supplementary draft differed from the archive: Table S11, 2022 / Blocks + buffer / Log / Centre RF, has the interval [−0.634, 1.010]. The release uses the archived value.
- The current packaging check did not refit the neural networks or execute a new Earth Engine export. The original runners, frozen memberships and archived fit verification results are retained.
