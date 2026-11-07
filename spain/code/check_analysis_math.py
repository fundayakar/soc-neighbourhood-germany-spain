"""Check block-bootstrap metric algebra against explicit resampled sklearn metrics."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from analyze import block_sums


def run(results,heterogeneity,out):
    p=pd.read_csv(Path(results)/'point_predictions_PRIVATE.csv',dtype={'point_id':str})
    h=pd.read_csv(heterogeneity,dtype={'point_id':str}).set_index('point_id').sort_index()
    y=p.drop_duplicates('point_id').set_index('point_id').actual_SOC.loc[h.index].to_numpy()
    _,code=np.unique(h.block,return_inverse=True);b=int(code.max()+1)
    w=np.random.default_rng(20261101).multinomial(b,np.ones(b)/b,size=2000)[0]
    sample=np.concatenate([np.tile(np.flatnonzero(code==k),int(w[k])) for k in range(b)])
    g=p[p.protocol.eq('block_100')&p.target.eq('log')]
    values={m:np.stack([z[z.seed.eq(s)].set_index('point_id').pred_SOC.loc[h.index].to_numpy()
                         for s in sorted(z.seed.unique())]) for m,z in g.groupby('model')}
    A,B=values['original_replay'],values['center_tiled'];yy=np.log1p(y);a=np.log1p(A);bb=np.log1p(B)
    n=len(sample)
    direct={'log_r2':np.mean([r2_score(yy[sample],a[s,sample])-r2_score(yy[sample],bb[s,sample]) for s in range(5)]),
            'raw_rmse':np.mean([np.sqrt(mean_squared_error(y[sample],A[s,sample]))-np.sqrt(mean_squared_error(y[sample],B[s,sample])) for s in range(5)]),
            'raw_mae':np.mean([mean_absolute_error(y[sample],A[s,sample])-mean_absolute_error(y[sample],B[s,sample]) for s in range(5)])}
    variance=(w@block_sums(yy**2,code,b)[0])/n-((w@block_sums(yy,code,b)[0])/n)**2
    mse_a=w@block_sums((A-y)**2,code,b).T/n;mse_b=w@block_sums((B-y)**2,code,b).T/n
    fast={'log_r2':float(np.mean(w@block_sums((bb-yy)**2-(a-yy)**2,code,b).T/n)/variance),
          'raw_rmse':float(np.mean(np.sqrt(mse_a)-np.sqrt(mse_b))),
          'raw_mae':float(np.mean(w@block_sums(np.abs(A-y)-np.abs(B-y),code,b).T/n))}
    delta={k:abs(float(direct[k])-fast[k]) for k in direct}
    assert max(delta.values())<1e-10
    x=h.ndvi_neighbour_std.to_numpy();x=(x-np.median(x))/(np.percentile(x,75)-np.percentile(x,25))
    gain=((bb-yy)**2-(a-yy)**2).mean(axis=0)
    direct_slope=np.linalg.lstsq(np.column_stack([np.ones(n),x[sample]]),gain[sample],rcond=None)[0][1]
    sx=w@block_sums(x,code,b)[0];sx2=w@block_sums(x*x,code,b)[0]
    sy=w@block_sums(gain,code,b)[0];sxy=w@block_sums(x*gain,code,b)[0]
    fast_slope=(sxy-sx*sy/n)/(sx2-sx*sx/n)
    assert abs(direct_slope-fast_slope)<1e-10
    status={'passed':True,'metric_deltas_from_explicit_resampling':delta,
            'slope_delta_from_explicit_OLS':float(abs(direct_slope-fast_slope)),
            'method':'one predetermined actual-data block draw; independent sklearn metrics and least-squares cross-check',
            'bootstrap_points_in_this_draw':n}
    Path(out).write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--heterogeneity',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.results,a.heterogeneity,a.output)
