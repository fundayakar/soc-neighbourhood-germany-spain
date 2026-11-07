# SOC neighbourhood prediction in Germany and Spain

Companion code and results for **Separating the contributions of neighbouring values and spatial arrangement to soil organic carbon prediction in Germany and Spain**, by Funda Yakar.

The experiments use separate LUCAS 2018 topsoil cohorts (Germany: 663; Spain: 3,754). Both cohorts use 2018 predictors for the main comparison. The German predictor-year sensitivity uses 2022 predictors against the same 2018 SOC. The German GPS sensitivity uses 662 paired observations and does not repeat the matched CNN input controls. Models were fitted and evaluated separately in each country.

The complete v1.0.0 reproducibility package, including the unlabelled predictor arrays, is archived on Zenodo: [10.5281/zenodo.24044760](https://doi.org/10.5281/zenodo.24044760).

## Reproduce the reported tables

```bash
python verify_release.py --source-only
python reproduce_tables.py --output ../soc_tables
```

This requires only Python 3.12 and the archived aggregate results. It reconstructs main Tables 2–3 and Supplementary Tables S10–S14 without training or access to individual SOC values. Results use the original fixed held-out predictions and design. A fresh fit on another software or hardware environment can select different epochs. See [reproducibility/README.md](reproducibility/README.md) for the preserved German component.

## Refit with independently licensed labels

Use Python 3.12 and the recorded Linux CPU dependencies:

```bash
python -m venv ../soc-venv
source ../soc-venv/bin/activate
python -m pip install -r requirements.txt
```

Run `python verify_release.py` to check every file in the complete deposit. The complete deposit includes unlabelled predictor arrays; the GitHub source snapshot omits the large arrays. If using the source snapshot, restore the NPZ files from the complete deposit to the same relative paths. Obtain the original LUCAS 2018 topsoil data independently from [ESDAC](https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data). Use the 0–20 cm SOC concentration records in g/kg. Label attachment checks the exact frozen SOC array hashes. If the file lacks a depth column, supply an independently verified 0–20 cm file. Keep labelled arrays and all new model outputs outside the repository.

```bash
# Germany: archived portable entry point and frozen folds
python reproducibility/attach_labels.py --labels /path/to/DE_labels.csv --output ../soc_local/DE
python reproducibility/train_release.py --data ../soc_local/DE --output ../soc_local/de_smoke --suite all --smoke
python reproducibility/train_release.py --data ../soc_local/DE --output ../soc_local/de_full --suite all

# Spain: both training regimes use fixed, archived roles
python spain/attach_labels.py --labels /path/to/ES_labels.csv --output ../soc_local/es_2018.npz
python spain/code/run_models.py --repo . --data ../soc_local/es_2018.npz --roles spain/data/roles/full/roles.csv --output ../soc_local/es_smoke --smoke
python spain/code/run_models.py --repo . --data ../soc_local/es_2018.npz --roles spain/data/roles/full/roles.csv --output ../soc_local/es_full
python spain/code/run_models.py --repo . --data ../soc_local/es_2018.npz --roles spain/data/roles/reduced/roles.csv --output ../soc_local/es_reduced

# Recompute Spanish block-bootstrap analyses on the new private outputs
python spain/code/analyze.py --results ../soc_local/es_full --heterogeneity spain/data/ndvi_block_metadata.csv --output ../soc_local/es_analysis_full
python spain/code/analyze.py --results ../soc_local/es_reduced --heterogeneity spain/data/ndvi_block_metadata.csv --output ../soc_local/es_analysis_reduced
python spain/code/compare_training_size.py --full ../soc_local/es_full --matched ../soc_local/es_reduced --heterogeneity spain/data/ndvi_block_metadata.csv --output ../soc_local/es_training_size
```

The Spanish full and reduced runs each comprise 400 model–fold–target fits. The reduced sets use the same Spanish test observations and match the German training count for the corresponding split. The original Spanish runner is preserved and resolves its five core modules from the bundled German source. Archived role files are the replay input; the historical role-generation utilities additionally require the original cohort metadata. The Spanish NPZ retains the exact model inputs and ID order but omits SOC and source/extraction coordinates. The NDVI table contains the non-centre NDVI standard deviation and coarse block IDs required for analysis. The German exploratory smearing and loss-gate analyses retain their historical source and aggregates; the portable training commands do not refit those pilots.

The original Spanish Earth Engine extraction scripts are in [spain/extraction](spain/extraction/README.md). Local refitting uses the archived arrays and does not require an Earth Engine account.

## Redistribution and provenance

The repository and deposit exclude source SOC observations, pointwise SOC predictions and residuals, private coordinate crosswalks, and fitted model checkpoints. See the [German component's data access notes](reproducibility/DATA_ACCESS.md) for the original sources and rights. Author-created code, documentation and aggregate results follow the [CC BY 4.0 notice](reproducibility/LICENSE.md). Third-party data retain their original conditions. The old Germany-only GitHub repository is not a dependency at runtime and is not modified by this release.
