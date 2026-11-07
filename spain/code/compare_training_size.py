"""Paired full-versus-Germany-size training comparison on unchanged Spanish tests."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import block_sums


def load(path):
    path=Path(path)
    c=json.loads((path/'completion.json').read_text());assert c['scientific_run_complete'] and c['configurations_completed']==400
    p=pd.read_csv(path/'point_predictions_PRIVATE.csv',dtype={'point_id':str})
    assert len(p)==300320
    return p


def run(full,matched,heterogeneity,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    A=load(full);B=load(matched)
    keys=['protocol','seed','fold','target','model','point_id']
    check=A[keys+['actual_SOC']].merge(B[keys+['actual_SOC']],on=keys,suffixes=('_full','_matched'),validate='one_to_one')
    assert len(check)==len(A)==len(B) and np.array_equal(check.actual_SOC_full,check.actual_SOC_matched)
    h=pd.read_csv(heterogeneity,dtype={'point_id':str}).set_index('point_id').sort_index();ids=h.index
    y=A.drop_duplicates('point_id').set_index('point_id').actual_SOC.loc[ids].to_numpy()
    _,code=np.unique(h.block,return_inverse=True);nblocks=int(code.max()+1);counts=np.bincount(code,minlength=nblocks)
    draws=np.random.default_rng(20261101).multinomial(nblocks,np.ones(nblocks)/nblocks,size=2000)
    n=draws@counts;log_y=np.log1p(y)
    log_var=(draws@block_sums(log_y**2,code,nblocks)[0])/n-((draws@block_sums(log_y,code,nblocks)[0])/n)**2
    scores={label:pd.read_csv(Path(path)/'pooled_oof_scores.csv') for label,path in [('full',full),('matched',matched)]}
    effects=[];interactions=[];tail_rows=[]
    for (protocol,target),ga in A.groupby(['protocol','target']):
        gb=B[B.protocol.eq(protocol)&B.target.eq(target)]
        seeds=sorted(ga.seed.unique());pred={}
        for label,g in [('full',ga),('matched',gb)]:
            pred[label]={model:np.stack([z[z.seed.eq(s)].set_index('point_id').pred_SOC.loc[ids].to_numpy() for s in seeds])
                         for model,z in g.groupby('model')}
        for model in pred['full']:
            aa=pred['full'][model];bb=pred['matched'][model]
            raw_se_a=(aa-y)**2;raw_se_b=(bb-y)**2
            log_se_a=(np.log1p(aa)-log_y)**2;log_se_b=(np.log1p(bb)-log_y)**2
            raw_a=(draws@block_sums(raw_se_a,code,nblocks).T)/n[:,None]
            raw_b=(draws@block_sums(raw_se_b,code,nblocks).T)/n[:,None]
            ae=(draws@block_sums(np.abs(aa-y)-np.abs(bb-y),code,nblocks).T)/n[:,None]
            ds=(draws@block_sums(log_se_b-log_se_a,code,nblocks).T)/n[:,None]
            distributions={'log_r2':ds.mean(axis=1)/log_var,
                           'raw_rmse':(np.sqrt(raw_a)-np.sqrt(raw_b)).mean(axis=1),
                           'raw_mae':ae.mean(axis=1)}
            row={'protocol':protocol,'target':target,'model':model,'contrast':'full minus Germany-size training'}
            for metric,vals in distributions.items():
                totals=[]
                for regime in ['full','matched']:
                    s=scores[regime];z=s[s.protocol.eq(protocol)&s.target.eq(target)&s.model.eq(model)]
                    totals.append(float(z[metric].mean()))
                lo,hi=np.percentile(vals,[2.5,97.5])
                row[metric+'_delta']=totals[0]-totals[1];row[metric+'_ci_low']=float(lo);row[metric+'_ci_high']=float(hi)
            effects.append(row)
            for regime,pp in [('full',aa),('matched',bb)]:
                se=(pp-y)**2;hi=y>=150
                tail_rows.append({'regime':regime,'protocol':protocol,'target':target,'model':model,
                                  'n_high_SOC':int(hi.sum()),'n_total':len(y),
                                  'mean_physical_SSE_share_SOC_ge150':float((se[:,hi].sum(axis=1)/se.sum(axis=1)).mean())})
        if target=='log':
            for control in ['center_tiled','shuffle_314159']:
                gains={}
                for regime in ['full','matched']:
                    original=pred[regime]['original_replay'];reference=pred[regime][control]
                    gains[regime]=((np.log1p(reference)-log_y)**2-(np.log1p(original)-log_y)**2).mean(axis=0)
                difference=gains['full']-gains['matched']
                vals=(draws@block_sums(difference,code,nblocks)[0])/n
                lo,hi=np.percentile(vals,[2.5,97.5]);r2_vals=vals/log_var;rlo,rhi=np.percentile(r2_vals,[2.5,97.5])
                interactions.append({'protocol':protocol,'control':control,
                    'full_log_MSE_gain':float(gains['full'].mean()),'matched_log_MSE_gain':float(gains['matched'].mean()),
                    'gain_change_full_minus_matched':float(difference.mean()),'ci_low':float(lo),'ci_high':float(hi),
                    'log_R2_gain_change':float(difference.mean()/np.var(log_y)),
                    'log_R2_change_ci_low':float(rlo),'log_R2_change_ci_high':float(rhi)})
    pd.DataFrame(effects).to_csv(out/'training_size_effects.csv',index=False)
    pd.DataFrame(interactions).to_csv(out/'training_size_context_interactions.csv',index=False)
    pd.DataFrame(tail_rows).to_csv(out/'high_SOC_error_shares.csv',index=False)
    status={'complete':True,'full_configurations':400,'matched_configurations':400,
            'same_Spanish_test_rows_verified':len(check),'n':3754,'blocks':nblocks,
            'bootstrap_draws':2000,'seed':20261101,
            'interpretation':'training-size effect within Spain under fixed algorithms; one frozen random subset per fold; not a causal country comparison',
            'uncertainty':'paired cluster bootstrap conditional on frozen models/predictions; no refits or multiplicity adjustment'}
    (out/'comparison_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(pd.DataFrame(effects).to_string(index=False));print(pd.DataFrame(interactions).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',required=True);p.add_argument('--matched',required=True)
    p.add_argument('--heterogeneity',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(a.full,a.matched,a.heterogeneity,a.output)
