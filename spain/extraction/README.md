# Spanish Earth Engine extraction

Local model replay starts from the frozen unlabelled arrays. Earth Engine is only required to rebuild the source exports.

1. Obtain LUCAS 2018 topsoil records independently under the ESDAC terms and upload the authorized source table to your own Earth Engine project.
2. Set the asset path in `GEE_ES_2018_coverage.js`. Its field names correspond to the study's imported asset. Check your asset schema before submitting the export task.
3. Run the coverage export. Use `check_coverage.py` with the authorized source CSV and coverage CSV to verify the retained cohort.
4. Append `GEE_ES_2018_patch_export_APPEND.js` to that Earth Engine script. The four export tasks use the study's frozen coverage exclusions.
5. Use `prepare_es_patches.py --help` to join the four exports to the audited cohort and validate the patch arrays.

These scripts create licensed, SOC-bearing exports and local arrays. Save those outputs outside the repository and do not add them to the public deposit. The coverage script changes the original asset path to a placeholder. Archived folds are used for the reported refits.
