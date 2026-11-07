"""Freeze Spanish spatial roles before model fitting; no SOC-based split selection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(data,cohort,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=dict(np.load(data,allow_pickle=False));ids=d['point_id'];xy=d['xy']
    assert len(ids)==3754 and d['X'].shape==(3754,14,17,17) and (d['valid']==1).all()
    ref=pd.read_csv(cohort,dtype={'point_id':str}).set_index('point_id').loc[ids]
    assert np.array_equal(d['y'],ref.OC.to_numpy())
    bxy=np.floor(xy/100000).astype(int)
    groups=np.array([f'{a}_{b}' for a,b in bxy]);unique=np.unique(groups)
    cells=ref.era_grid_x.astype(str)+'_'+ref.era_grid_y.astype(str)
    rows=[];diagnostics=[]
    for protocol,repeats in [('block_100',5),('block_100_buffer20',3)]:
        for seed in range(repeats):
            assignment=dict(zip(unique,np.random.RandomState(seed).randint(0,5,len(unique))))
            folds=np.array([assignment[g] for g in groups])
            for fold in range(5):
                test=np.flatnonzero(folds==fold);train=np.flatnonzero(folds!=fold)
                excluded=np.array([],dtype=int)
                if protocol.endswith('buffer20'):
                    ok=cKDTree(xy[test]).query(xy[train])[0]>20000
                    excluded=train[~ok];train=train[ok]
                assert not set(groups[train])&set(groups[test])
                ia,ib=next(GroupKFold(3).split(train,groups=groups[train]))
                it,iv=train[ia],train[ib];inner_excluded=np.array([],dtype=int)
                if protocol.endswith('buffer20'):
                    ok=cKDTree(xy[iv]).query(xy[it])[0]>20000
                    inner_excluded=it[~ok];it=it[ok]
                assert min(len(train),len(test),len(it),len(iv))>=20
                assert set(it)|set(iv)|set(inner_excluded)==set(train)
                assert not set(groups[it])&set(groups[iv])
                distance=float(cKDTree(xy[train]).query(xy[test])[0].min())
                inner_distance=float(cKDTree(xy[it]).query(xy[iv])[0].min())
                patch_distance=float(cKDTree(d['patch_center_xy'][train]).query(d['patch_center_xy'][test])[0].min())
                assert patch_distance>np.sqrt(2)*340
                if protocol.endswith('buffer20'):assert min(distance,inner_distance)>20000
                meta=dict(protocol=protocol,seed=seed,fold=fold)
                diagnostics.append(dict(**meta,n_train=len(train),n_test=len(test),
                    n_epoch_train=len(it),n_epoch_valid=len(iv),n_buffer_excluded=len(excluded),
                    n_inner_buffer_excluded=len(inner_excluded),n_train_blocks=len(np.unique(groups[train])),
                    n_test_blocks=len(np.unique(groups[test])),train_test_min_m=distance,
                    inner_train_valid_min_m=inner_distance,patch_train_test_min_m=patch_distance,
                    n_test_sharing_native_climate_cell=int(np.isin(cells.iloc[test],cells.iloc[train]).sum())))
                for role,indices in [('train',train),('test',test),('epoch_train',it),
                                     ('epoch_valid',iv),('buffer_excluded',excluded),
                                     ('epoch_buffer_excluded',inner_excluded)]:
                    rows.extend(dict(**meta,role=role,point_id=ids[i]) for i in indices)
    roles=pd.DataFrame(rows);diag=pd.DataFrame(diagnostics)
    for (p,s),g in roles[roles.role.eq('test')].groupby(['protocol','seed']):
        assert len(g)==3754 and g.point_id.nunique()==3754
    roles.to_csv(out/'roles.csv',index=False);diag.to_csv(out/'fold_diagnostics.csv',index=False)
    ndvi=np.delete(d['X'][:,6].reshape(3754,289),144,axis=1).astype('float64')
    pd.DataFrame({'point_id':ids,'ndvi_neighbour_std':ndvi.std(axis=1,ddof=0),'block':groups})
    hetero=pd.DataFrame({'point_id':ids,'ndvi_neighbour_std':ndvi.std(axis=1,ddof=0),'block':groups})
    hetero.to_csv(out/'heterogeneity_PRIVATE.csv',index=False)
    report={'n':3754,'blocks':len(unique),'fold_conditions':len(diag),
            'protocol_repeats':{'block_100':5,'block_100_buffer20':3},
            'all_role_and_distance_checks_passed':True,'data_sha256':sha(data),
            'roles_sha256':sha(out/'roles.csv'),'source_sha256':sha(__file__),
            'heterogeneity':'population standard deviation of 288 non-centre NDVI values; no fitted scale or SOC used',
            'ranges':{p:{c:{'min':float(g[c].min()),'max':float(g[c].max())} for c in
                        ['n_train','n_test','n_epoch_train','n_epoch_valid','train_test_min_m','inner_train_valid_min_m']}
                      for p,g in diag.groupby('protocol')},'models_fitted':False}
    (out/'role_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--cohort',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.data,a.cohort,a.output),indent=2))
