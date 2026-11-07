"""Scientific figures from complete paired-control outputs; no fit selection."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(analysis):
    out=Path(analysis);q=pd.read_csv(out/'heterogeneity_quartiles.csv')
    e=pd.read_csv(out/'paired_contrasts.csv')
    protocols=['block_100','block_100_buffer20']
    titles=['100 km blocks','100 km blocks + 20 km buffer']
    colours={'original_vs_centre':'#2463a6','original_vs_shuffled':'#c75b20'}
    labels={'original_vs_centre':'Original versus centre-only','original_vs_shuffled':'Original versus shuffled'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
    for ax,p,title in zip(axes,protocols,titles):
        ax.axhline(0,color='0.45',lw=.8)
        for label in colours:
            g=q[q.protocol.eq(p)&q.contrast.eq(label)].sort_values('quartile')
            x=g.quartile.to_numpy()+(-.035 if label=='original_vs_centre' else .035)
            y=g.mean_log_squared_error_reduction.to_numpy();c=colours[label]
            ax.vlines(x,g.ci_low,g.ci_high,color=c,lw=1.3)
            ax.plot(x,y,'o-',color=c,label=labels[label],ms=4,lw=1.1)
        ax.set(title=title,xlabel='Neighbourhood NDVI variability quartile',xticks=[1,2,3,4])
    axes[0].set_ylabel('Reduction in squared log1p error')
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Spain: predefined neighbourhood-heterogeneity comparison',fontsize=11)
    fig.tight_layout()
    fig.savefig(out/'Spain_heterogeneity.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/'Spain_heterogeneity.svg',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
    order=[(p,c) for p in protocols for c in colours]
    for ax,metric,title in zip(axes,['log_r2','raw_rmse'],['Log-scale R² difference','Physical-unit RMSE difference (g/kg)']):
        ax.axvline(0,color='0.45',lw=.8)
        for i,(p,c) in enumerate(order):
            row=e[e.protocol.eq(p)&e.target.eq('log')&e.contrast.eq(c)].iloc[0]
            lo=row[metric+'_ci_low'];hi=row[metric+'_ci_high'];v=row[metric+'_delta']
            ax.hlines(i,lo,hi,color=colours[c],lw=1.5);ax.plot(v,i,'o',color=colours[c],ms=5)
        ax.set_xlabel(title+'\n(original minus control)')
        ax.set_yticks(range(4))
        ax.set_yticklabels(['Blocks: centre-only','Blocks: shuffled','Buffer: centre-only','Buffer: shuffled'])
    axes[0].invert_yaxis()
    fig.suptitle('Spain: matched CNN contrasts under log-target training',fontsize=11)
    fig.tight_layout()
    fig.savefig(out/'Spain_matched_contrasts.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/'Spain_matched_contrasts.svg',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);a=p.parse_args();run(a.analysis)
