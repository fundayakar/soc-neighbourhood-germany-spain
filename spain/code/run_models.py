"""Matched-capacity Spanish replication. Private outputs, frozen roles, resumable fits."""
import argparse,concurrent.futures as cf,hashlib,json,multiprocessing as mp,os,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
from sklearn.ensemble import RandomForestRegressor

G={}
NEURAL=['original_replay','center_tiled','shuffle_314159']
MODELS=NEURAL+['centerRF','contextRF']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def modules(repo):
    root=Path(repo)/'reproducibility/experiments'
    sys.path[:0]=[str(root/'main/cnn'),str(root/'main'),str(root/'main/stage2'),str(root/'cnn_controls')]
    import train_patch_models as t
    import run_ablations as c
    import rf_compare
    t.torch.set_num_threads(1)
    return t,c,rf_compare


def initialize(repo,data,roles,out,fingerprint,epochs):
    t,c,rf=modules(repo)
    original=dict(np.load(data,allow_pickle=False))
    transformed={v:c.transformed(original,v) for v in NEURAL}
    for d in transformed.values():
        assert np.array_equal(d['X'][:,:,8,8],original['X'][:,:,8,8])
    role_table=pd.read_csv(roles,dtype={'point_id':str})
    idx={p:i for i,p in enumerate(original['point_id'])}
    splits={}
    for key,g in role_table.groupby(['protocol','seed','fold']):
        splits[key]={role:np.array(sorted(idx[p] for p in rows.point_id),dtype=int)
                     for role,rows in g.groupby('role')}
    rf_x={v:t.rf_features(original,v) for v in ['centerRF','contextRF']}
    G.update(t=t,c=c,rf=rf,original=original,transformed=transformed,splits=splits,
             out=Path(out),fingerprint=fingerprint,epochs=epochs,rf_x=rf_x)


def job(spec):
    protocol,repeat,fold,target,model=spec
    name=f'{protocol}_s{repeat}_f{fold}_{target}_{model}'
    path=G['out']/'runs'/f'{name}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['fingerprint']==G['fingerprint']
        if model in NEURAL:assert path.with_suffix('.pt').exists()
        return name,0,'cached'
    t,c,orig=G['t'],G['c'],G['original']
    roles=G['splits'][(protocol,repeat,fold)]
    tr,te,it,iv=[roles[k] for k in ['train','test','epoch_train','epoch_valid']]
    assert not set(tr)&set(te) and set(it)<=set(tr) and set(iv)<=set(tr) and not set(it)&set(iv)
    started=time.time();epoch=None;curve=[];netseed=None
    if model in NEURAL:
        data=G['transformed'][model];netseed=42+100*repeat+fold
        _,_,epoch,curve=c.fit(t,orig,data,it,iv,target,netseed,G['epochs'],True)
        trained,norm,_,_=c.fit(t,orig,data,tr,None,target,netseed,epoch,False)
        assert sum(p.numel() for p in trained.parameters())==8449
        pred=t.neural_predict(trained,norm,data,te,target,'cpu')
        checkpoint=dict(state_dict=trained.state_dict(),normalization=norm,epochs=epoch,
                        network_seed=netseed,fingerprint=G['fingerprint'],variant=model,parameter_count=8449)
        temp=path.with_suffix('.pt.tmp');t.torch.save(checkpoint,temp);temp.replace(path.with_suffix('.pt'))
    else:
        y=np.log1p(orig['y']) if target=='log' else orig['y']
        fitted=RandomForestRegressor(**G['rf'].RF).fit(G['rf_x'][model][tr],y[tr])
        pred=fitted.predict(G['rf_x'][model][te])
        pred=np.maximum(0,np.expm1(pred) if target=='log' else pred)
    assert len(pred)==len(te) and np.isfinite(pred).all()
    r=dict(protocol=protocol,seed=repeat,fold=fold,target=target,model=model,
           point_id=orig['point_id'][te].tolist(),actual_SOC=orig['y'][te].tolist(),pred_SOC=pred.tolist(),
           network_seed=netseed,selected_epochs=epoch,inner_curve=curve,
           metrics=t.raw_metrics(orig['y'][te],pred),seconds=time.time()-started,fingerprint=G['fingerprint'])
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(r)+'\n');temp.replace(path)
    return name,round(r['seconds'],1),'completed'


def summarize(out,jobs,fingerprint,t):
    frames=[];fold_rows=[]
    for protocol,repeat,fold,target,model in jobs:
        name=f'{protocol}_s{repeat}_f{fold}_{target}_{model}'
        r=json.loads((out/'runs'/f'{name}.json').read_text());assert r['fingerprint']==fingerprint
        meta={k:r[k] for k in ['protocol','seed','fold','target','model']}
        frames.append(pd.DataFrame(dict(**meta,point_id=r['point_id'],actual_SOC=r['actual_SOC'],pred_SOC=r['pred_SOC'])))
        fold_rows.append(dict(**meta,epochs=r['selected_epochs'],seconds=r['seconds'],**r['metrics']))
    p=pd.concat(frames,ignore_index=True);p.to_csv(out/'point_predictions_PRIVATE.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_scores.csv',index=False)
    pooled=[]
    for key,g in p.groupby(['protocol','seed','target','model']):
        if len(g)!=3754:continue
        assert g.point_id.nunique()==3754
        pooled.append(dict(zip(['protocol','seed','target','model'],key),**t.raw_metrics(g.actual_SOC.to_numpy(),g.pred_SOC.to_numpy())))
    if pooled:
        s=pd.DataFrame(pooled);s.to_csv(out/'pooled_oof_scores.csv',index=False)
        s.groupby(['protocol','target','model'])[['log_r2','raw_r2','raw_rmse','raw_mae']].mean().to_csv(out/'mean_scores.csv')


def main(a):
    repo=a.repo.resolve();out=a.output.resolve();data=a.data.resolve();roles=a.roles.resolve()
    if repo==out or repo in out.parents:raise ValueError('Labelled outputs must remain outside the repository.')
    t,c,rf=modules(repo)
    with np.load(data,allow_pickle=False) as d:
        assert d['X'].shape==(3754,14,17,17) and (d['valid']==1).all()
    jobs=[(protocol,repeat,fold,target,model)
          for protocol,repeats in [('block_100',5),('block_100_buffer20',3)]
          for repeat in range(repeats) for fold in range(5)
          for target in ['log','raw'] for model in MODELS]
    if a.smoke:jobs=[j for j in jobs if j[0]=='block_100' and j[1]==0 and j[2]==0 and j[3]=='log']
    epochs=4 if a.smoke else 100
    sources=[Path(__file__),Path(__file__).with_name('prepare_roles.py'),
             repo/'reproducibility/experiments/main/cnn/train_patch_models.py',
             repo/'reproducibility/experiments/cnn_controls/run_ablations.py',
             repo/'reproducibility/experiments/main/target_compare.py',
             repo/'reproducibility/experiments/main/cnn/prepare_patches.py',
             repo/'reproducibility/experiments/main/stage2/rf_compare.py']
    design=dict(name='Spain_2018_matched_CNN_geographic_replication',n=3754,
        technical_smoke=a.smoke,max_epochs=epochs,patience=15,jobs=jobs,
        data_sha256=sha(data),roles_sha256=sha(roles),RF=rf.RF,
        source_sha256={str(p):sha(p) for p in sources},
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                      scipy=scipy.__version__,sklearn=sklearn.__version__,torch=t.torch.__version__),
        architecture='Existing 8449-parameter CNN, common original training-patch normalization across inputs',
        network_seed='42+100*outer_repeat+fold; one matched initialization per fold; no initialization search',
        epoch_selection='Frozen inner grouped split; separate per input and target; fresh fit on outer training',
        targets=['raw','log1p'],prediction='expm1 for log fits; nonnegative clipping, no smearing or upper cap',
        primary_heterogeneity_loss='squared log1p prediction error under log-target training',
        permutation_seed=314159,input_source_library_id='libfile_44e78b06d7848191adb84bee4454ff5b',
        interpretation='Geographic replication within the same LUCAS campaign; not independent laboratory validation')
    fingerprint=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();design['fingerprint']=fingerprint
    out.mkdir(parents=True,exist_ok=True);(out/'runs').mkdir(exist_ok=True)
    if (out/'design.json').exists():assert json.loads((out/'design.json').read_text())==json.loads(json.dumps(design))
    else:(out/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    print(f'Frozen configurations: {len(jobs)}; smoke={a.smoke}',flush=True)
    with cf.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),
        initializer=initialize,initargs=(repo,data,roles,out,fingerprint,epochs)) as pool:
        futures=[pool.submit(job,j) for j in jobs]
        for i,f in enumerate(cf.as_completed(futures),1):
            name,seconds,status=f.result();print(i,len(jobs),name,seconds,status,flush=True)
    summarize(out,jobs,fingerprint,t)
    (out/'completion.json').write_text(json.dumps(dict(configurations_completed=len(jobs),
        scientific_run_complete=not a.smoke,technical_smoke_complete=a.smoke,fingerprint=fingerprint),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--roles',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
