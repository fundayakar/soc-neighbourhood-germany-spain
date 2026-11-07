"""Match Spanish training counts to Germany while keeping Spanish tests fixed."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(data,spanish_roles,german_roles,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=dict(np.load(data,allow_pickle=False));ids=d['point_id'];xy=d['xy']
    idx={p:i for i,p in enumerate(ids)}
    full=pd.read_csv(spanish_roles,dtype={'point_id':str})
    de=pd.read_csv(german_roles,dtype={'point_id':str})
    counts=de[de.role.eq('train')].groupby(['protocol','seed','fold']).size()
    groups=np.array([f'{a}_{b}' for a,b in np.floor(xy/100000).astype(int)])
    rows=[];stats=[]
    for key,g in full.groupby(['protocol','seed','fold']):
        protocol,seed,fold=key;n=int(counts.loc[key])
        tr=np.array(sorted(idx[p] for p in g[g.role.eq('train')].point_id))
        te=np.array(sorted(idx[p] for p in g[g.role.eq('test')].point_id))
        subset_seed=20261101+100*int(seed)+int(fold)
        small=np.sort(np.random.default_rng(subset_seed).choice(tr,size=n,replace=False))
        a,b=next(GroupKFold(3).split(small,groups=groups[small]));it,iv=small[a],small[b]
        inner_excluded=np.array([],dtype=int)
        if protocol.endswith('buffer20'):
            ok=cKDTree(xy[iv]).query(xy[it])[0]>20000
            inner_excluded=it[~ok];it=it[ok]
        assert min(len(small),len(te),len(it),len(iv))>=20
        assert not set(groups[small])&set(groups[te]) and not set(groups[it])&set(groups[iv])
        assert set(it)|set(iv)|set(inner_excluded)==set(small)
        distance=float(cKDTree(xy[small]).query(xy[te])[0].min())
        inner_distance=float(cKDTree(xy[it]).query(xy[iv])[0].min())
        if protocol.endswith('buffer20'):assert min(distance,inner_distance)>20000
        meta=dict(protocol=protocol,seed=int(seed),fold=int(fold))
        stats.append(dict(**meta,full_spanish_train=len(tr),matched_train=len(small),german_train=n,
                          n_test=len(te),n_epoch_train=len(it),n_epoch_valid=len(iv),
                          train_test_min_m=distance,inner_train_valid_min_m=inner_distance,subset_seed=subset_seed))
        for role,ix in [('train',small),('test',te),('epoch_train',it),('epoch_valid',iv),
                        ('epoch_buffer_excluded',inner_excluded)]:
            rows.extend(dict(**meta,role=role,point_id=ids[i]) for i in ix)
    result=pd.DataFrame(rows);result.to_csv(out/'roles.csv',index=False)
    pd.DataFrame(stats).to_csv(out/'fold_diagnostics.csv',index=False)
    for key,g in full[full.role.eq('test')].groupby(['protocol','seed','fold']):
        small=result[result.protocol.eq(key[0])&result.seed.eq(key[1])&result.fold.eq(key[2])&result.role.eq('test')]
        assert set(g.point_id)==set(small.point_id)
    status={'n_spanish_test_cohort':3754,'fold_conditions':len(stats),
            'all_test_memberships_identical_to_full_spanish_run':True,
            'training_counts_match_Germany_exactly':True,'source_spanish_roles_sha256':sha(spanish_roles),
            'source_german_roles_sha256':sha(german_roles),'roles_sha256':sha(out/'roles.csv'),
            'sampling':'one fixed uniform training subset per protocol-repeat-fold; no held-out labels or scores used',
            'inner':'first 3-way GroupKFold on the sampled training pool; inner 20km buffer for buffered protocol',
            'source_sha256':sha(__file__),'models_fitted':False}
    (out/'role_validation.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--spanish-roles',required=True)
    p.add_argument('--german-roles',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(run(a.data,a.spanish_roles,a.german_roles,a.output),indent=2))
