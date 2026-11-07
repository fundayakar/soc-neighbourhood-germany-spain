"""Validate all four real Spanish patch CSVs against the frozen coverage cohort.

Writes licensed, PRIVATE labelled NPZ input only after all checks pass.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS=['B2','B3','B4','B8','B11','B12','NDVI','VV','VH','VV_minus_VH',
       'elev','slope','aspect_sin','aspect_cos']
CLIMATE=['tp_winter_m','t2m_summer_C','sm_annual']


def parse_array(value):
    if not isinstance(value,str):
        raise ValueError('Missing real raster array; tabular vectors are not patches.')
    try:
        a=json.loads(value)
    except json.JSONDecodeError:
        a=ast.literal_eval(value)
    a=np.asarray(a,dtype=np.float64)
    if a.shape!=(17,17) or not np.isfinite(a).all():
        raise ValueError('Expected finite 17x17 array.')
    return a


def run(parts,cohort,out):
    out=Path(out)
    ref=pd.read_csv(cohort,dtype={'point_id':str,'POINTID':str}).set_index('point_id').sort_index()
    assert len(ref)==3754 and ref.index.is_unique
    frames=[];source_hashes={}
    for path in parts:
        path=Path(path)
        frames.append(pd.read_csv(path,dtype={'point_id':str,'POINTID':str}))
        source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    d=pd.concat(frames,ignore_index=True)
    assert len(d)==len(ref) and d.point_id.is_unique
    assert set(d.point_id)==set(ref.index),'Patch exports do not match frozen source IDs.'
    d=d.set_index('point_id').loc[ref.index]
    for c in ['POINTID','NUTS_0','Depth','SURVEY_DATE','LC','LC0_Desc','LC1_Desc']:
        assert d[c].equals(ref[c]),f'Metadata changed: {c}'
    assert d.predictor_year.eq(2018).all()
    for c in ['OC','source_lon','source_lat','source_x3035','source_y3035']:
        assert np.allclose(d[c],ref[c],rtol=0,atol=1e-8),f'Source data changed: {c}'
    X=np.stack([np.stack([parse_array(v) for v in d[b]]) for b in BANDS],axis=1)
    valid=np.stack([parse_array(v) for v in d.valid])[:,None]
    assert np.all(valid==1),'Frozen cohort must have 289/289 valid pixels; stop, do not silently drop rows.'
    clim=d[CLIMATE].to_numpy(float)
    assert np.isfinite(clim).all()
    assert np.allclose(clim,ref[CLIMATE],rtol=1e-7,atol=1e-8),'Native climate extraction changed.'
    center=X[:,:,8,8]
    reference=ref[BANDS].to_numpy(float)
    assert np.allclose(center,reference,rtol=1e-5,atol=1e-6),'Patch centre differs from coverage predictor vector.'
    xy=ref[['source_x3035','source_y3035']].to_numpy(float)
    px=d[['patch_center_x3035','patch_center_y3035']].to_numpy(float)
    assert np.isfinite(px).all()
    offset=np.linalg.norm(xy-px,axis=1)
    assert offset.max()<=np.sqrt(2)*10+.2,'Patch grid displaced relative to source points.'
    grid_remainder=np.remainder(px,20)
    assert np.allclose(grid_remainder,10,atol=.2,rtol=0),'Unexpected 20m grid phase.'
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'ES_main_2018_PRIVATE.npz',
        X=X.astype('float32'),valid=valid.astype('float32'),
        climate=clim.astype('float32'),y=ref.OC.to_numpy('float64'),
        point_id=ref.index.to_numpy(str),xy=xy,patch_center_xy=px,
        year=np.array(2018),bands=np.array(BANDS),climate_names=np.array(CLIMATE))
    report={'rows':len(d),'array_shape':list(X.shape),'valid_shape':list(valid.shape),
            'all_masks_complete':True,'centre_parity_passed':True,
            'max_center_offset_m':float(offset.max()),
            'max_center_abs_delta':float(np.max(np.abs(center-reference))),
            'native_climate_parity_passed':True,'input_hashes':source_hashes,
            'cohort_sha256':hashlib.sha256(Path(cohort).read_bytes()).hexdigest(),
            'cnn_trained':False}
    (out/'patch_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts',nargs=4,required=True)
    p.add_argument('--cohort',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(run(a.parts,a.cohort,a.output),indent=2))
