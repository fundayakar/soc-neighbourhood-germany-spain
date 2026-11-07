"""Prespecified paired-block summaries of the completed Spanish replication."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

PAIRS=[('original_vs_centre','original_replay','center_tiled'),
       ('original_vs_shuffled','original_replay','shuffle_314159'),
       ('ContextRF_vs_CentreRF','contextRF','centerRF'),
       ('original_vs_ContextRF','original_replay','contextRF')]


def block_sums(a,code,b):
    return np.stack([np.bincount(code,weights=row,minlength=b) for row in np.atleast_2d(a)])


def run(results,heterogeneity,out):
    results=Path(results);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    completion=json.loads((results/'completion.json').read_text())
    assert completion['scientific_run_complete'] and completion['configurations_completed']==400
    p=pd.read_csv(results/'point_predictions_PRIVATE.csv',dtype={'point_id':str})
    assert len(p)==3754*8*2*5
    h=pd.read_csv(heterogeneity,dtype={'point_id':str}).set_index('point_id').sort_index()
    assert len(h)==3754 and h.index.is_unique
    y=p.drop_duplicates('point_id').set_index('point_id').actual_SOC.loc[h.index].to_numpy()
    unique,code=np.unique(h.block.to_numpy(),return_inverse=True);b=len(unique)
    assert b==69
    counts=np.bincount(code,minlength=b)
    draws=np.random.default_rng(20261101).multinomial(b,np.ones(b)/b,size=2000)
    sizes=draws@counts
    h_raw=h.ndvi_neighbour_std.to_numpy();iqr=np.percentile(h_raw,75)-np.percentile(h_raw,25)
    assert iqr>0
    hv=(h_raw-np.median(h_raw))/iqr
    quartiles=pd.qcut(h_raw,4,labels=False)
    score=pd.read_csv(results/'pooled_oof_scores.csv')
    effects=[];slopes=[];bins=[]
    for (protocol,target),g in p.groupby(['protocol','target']):
        seeds=sorted(g.seed.unique());preds={}
        for model,z in g.groupby('model'):
            preds[model]=np.stack([z[z.seed.eq(seed)].set_index('point_id').pred_SOC.loc[h.index].to_numpy() for seed in seeds])
            assert np.isfinite(preds[model]).all()
        for label,left,right in PAIRS:
            A,B=preds[left],preds[right]
            delta_samples={}
            for prefix,yy,aa,bb in [('log',np.log1p(y),np.log1p(A),np.log1p(B)),('raw',y,A,B)]:
                se_a=block_sums((aa-yy)**2,code,b);se_b=block_sums((bb-yy)**2,code,b)
                ms_a=draws@se_a.T/sizes[:,None];ms_b=draws@se_b.T/sizes[:,None]
                if prefix=='log':
                    sums=block_sums(yy,code,b)[0];sums2=block_sums(yy**2,code,b)[0]
                    variance=(draws@sums2)/sizes-((draws@sums)/sizes)**2
                    delta_samples['log_r2']=((ms_b-ms_a).mean(axis=1))/variance
                else:
                    delta_samples['raw_rmse']=(np.sqrt(ms_a)-np.sqrt(ms_b)).mean(axis=1)
                    ae_a=block_sums(np.abs(aa-yy),code,b);ae_b=block_sums(np.abs(bb-yy),code,b)
                    delta_samples['raw_mae']=((draws@(ae_a-ae_b).T)/sizes[:,None]).mean(axis=1)
            s=score[score.protocol.eq(protocol)&score.target.eq(target)]
            row={'protocol':protocol,'target':target,'contrast':label,'left':left,'right':right,'n_repeats':len(seeds)}
            for metric,values in delta_samples.items():
                row[metric+'_delta']=float(s[s.model.eq(left)][metric].mean()-s[s.model.eq(right)][metric].mean())
                lo,hi=np.percentile(values,[2.5,97.5]);row[metric+'_ci_low']=float(lo);row[metric+'_ci_high']=float(hi)
            effects.append(row)
            if target=='log' and label in ['original_vs_centre','original_vs_shuffled']:
                gain=((np.log1p(B)-np.log1p(y))**2-(np.log1p(A)-np.log1p(y))**2).mean(axis=0)
                # Positive gain favours original patches. Positive slope means greater
                # error reduction per one-IQR increase in neighbourhood NDVI variability.
                sh=block_sums(hv,code,b)[0];sh2=block_sums(hv**2,code,b)[0]
                sg=block_sums(gain,code,b)[0];shg=block_sums(hv*gain,code,b)[0]
                n=sizes;xh=draws@sh;xh2=draws@sh2;xg=draws@sg;xhg=draws@shg
                bs=(xhg-xh*xg/n)/(xh2-xh*xh/n)
                slope=float(np.sum((hv-hv.mean())*(gain-gain.mean()))/np.sum((hv-hv.mean())**2))
                lo,hi=np.percentile(bs,[2.5,97.5])
                slopes.append({'protocol':protocol,'contrast':label,'slope_per_NDVI_std_IQR':slope,
                               'ci_low':float(lo),'ci_high':float(hi),'NDVI_std_IQR':float(iqr),
                               'loss':'paired reduction in squared log1p error; positive favours original'})
                for q in range(4):
                    z=quartiles==q
                    cq=np.bincount(code[z],minlength=b);gq=np.bincount(code[z],weights=gain[z],minlength=b)
                    vals=(draws@gq)/(draws@cq)
                    lo,hi=np.percentile(vals,[2.5,97.5])
                    bins.append({'protocol':protocol,'contrast':label,'quartile':q+1,'n':int(z.sum()),
                                 'H_min':float(h_raw[z].min()),'H_max':float(h_raw[z].max()),
                                 'mean_log_squared_error_reduction':float(gain[z].mean()),
                                 'ci_low':float(lo),'ci_high':float(hi)})
    pd.DataFrame(effects).to_csv(out/'paired_contrasts.csv',index=False)
    pd.DataFrame(slopes).to_csv(out/'heterogeneity_slopes.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'heterogeneity_quartiles.csv',index=False)
    status={'complete':True,'n':3754,'configurations':400,'blocks':b,'bootstrap_draws':2000,
            'bootstrap_seed':20261101,'resampling':'paired fixed 100km blocks; conditional on trained OOF predictions; no refitting',
            'repetitions':'metrics averaged over pooled OOF scores per repeat; not an ensemble; repeats are not independent samples',
            'intervals':'percentile 95%, descriptive and not adjusted for multiple contrasts',
            'heterogeneity':'predefined non-centre NDVI standard deviation; predictive association, not a causal mechanism',
            'sample_size_matched_training':'not assessed by this within-regime summary; see separate training-size comparison',
            'France':'no predictor extraction or model fitting'}
    (out/'analysis_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(pd.DataFrame(effects).to_string(index=False))
    print(pd.DataFrame(slopes).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--heterogeneity',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.results,a.heterogeneity,a.output)
