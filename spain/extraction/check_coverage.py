"""Validate the Spanish coverage export against the frozen source metadata."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ['B2','B3','B4','B8','B11','B12','NDVI','VV','VH','VV_minus_VH',
         'elev','slope','aspect_sin','aspect_cos']
CLIMATE = ['tp_winter_m','t2m_summer_C','sm_annual']


def run(source, coverage, output):
    source, coverage, output = Path(source), Path(coverage), Path(output)
    s = pd.read_csv(source, dtype={'point_id':str,'POINTID':str}).set_index('point_id')
    d = pd.read_csv(coverage, dtype={'point_id':str,'POINTID':str}).set_index('point_id')
    assert len(s)==3763 and s.index.is_unique and d.index.is_unique
    assert set(d.index)==set(s.index), 'Source IDs changed or coverage rows are missing.'
    d = d.loc[s.index]
    for c in ['POINTID','NUTS_0','Depth','SURVEY_DATE','LC','LC0_Desc','LC1_Desc']:
        assert s[c].equals(d[c]), f'Metadata changed: {c}'
    for c in ['OC','source_lon','source_lat']:
        assert np.array_equal(s[c].to_numpy(),d[c].to_numpy()), f'Source changed: {c}'
    assert d.NUTS_0.eq('ES').all() and d.Depth.eq('0-20 cm').all()
    assert d.predictor_year.eq(2018).all()
    assert np.isfinite(d[['OC','source_lon','source_lat','source_x3035','source_y3035']]).all().all()
    assert d.OC.ge(0).all()
    for c in ['center_valid','patch_all_valid']:
        assert d[c].isin([0,1]).all()
    assert d.patch_valid_fraction.between(0,1).all()
    assert np.allclose(d.patch_valid_fraction*289,np.round(d.patch_valid_fraction*289),atol=1e-7,rtol=0)
    assert d.patch_all_valid.eq(1).equals(d.patch_valid_fraction.eq(1))
    assert np.allclose(d.VV_minus_VH,d.VV-d.VH,rtol=0,atol=1e-7)
    assert np.allclose(d.NDVI,(d.B8-d.B4)/(d.B8+d.B4),rtol=0,atol=1e-7)
    assert np.allclose(d.aspect_sin**2+d.aspect_cos**2,1,atol=1e-7,rtol=0)
    raster = np.isfinite(d[BANDS].to_numpy(float)).all(axis=1)
    clim = np.isfinite(d[CLIMATE].to_numpy(float)).all(axis=1)
    full = d.patch_all_valid.eq(1).to_numpy()
    keep = raster & clim & full & d.center_valid.eq(1).to_numpy()
    output.mkdir(parents=True,exist_ok=True)
    frozen = d.loc[keep].sort_index()
    frozen.to_csv(output/'ES_2018_frozen_cohort_PRIVATE.csv',index=True)
    rejected = d.loc[~keep].copy()
    rejected['missing_climate'] = ~clim[~keep]
    rejected['incomplete_patch'] = ~full[~keep]
    rejected.to_csv(output/'ES_2018_exclusions_PRIVATE.csv',index=True)
    blocks = pd.DataFrame({'x':np.floor(frozen.source_x3035/100000).astype(int),
                           'y':np.floor(frozen.source_y3035/100000).astype(int)})
    report = {
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'coverage_sha256':hashlib.sha256(coverage.read_bytes()).hexdigest(),
        'source_rows':len(s),'coverage_rows':len(d),'source_parity_passed':True,
        'raster_center_finite':int(raster.sum()),'complete_patch_rows':int(full.sum()),
        'finite_climate_rows':int(clim.sum()),'missing_climate_rows':int((~clim).sum()),
        'incomplete_patch_rows':int((~full).sum()),
        'climate_patch_failure_overlap':int((~clim & ~full).sum()),
        'retained_rows':len(frozen),'excluded_rows':len(rejected),
        'retention_percent':100*len(frozen)/len(d),
        'excluded_ids':sorted(rejected.index.tolist()),
        'retained_id_sha256':hashlib.sha256(('\n'.join(frozen.index)+'\n').encode()).hexdigest(),
        'retained_100km_blocks':len(blocks.drop_duplicates()),
        'retained_land_cover':frozen.LC0_Desc.value_counts().to_dict(),
        'excluded_land_cover':rejected.LC0_Desc.value_counts().to_dict(),
        'retained_soc_g_kg':{'min':float(frozen.OC.min()),'median':float(frozen.OC.median()),
                             'max':float(frozen.OC.max()),'n_ge_150':int(frozen.OC.ge(150).sum())},
        'sar_difference_ndvi_aspect_checks_passed':True,
        'eligibility_rule':'all 14 center rasters and 3 climate scalars finite; 289/289 common raster pixels valid; no imputation',
        'patch_array_export_status':'not yet run','model_training_status':'not started'
    }
    (output/'coverage_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--coverage',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(run(a.source,a.coverage,a.output),indent=2))
