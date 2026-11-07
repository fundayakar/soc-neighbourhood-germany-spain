"""Replay every saved Spanish CNN checkpoint and audit every model's test rows."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from run_models import modules,NEURAL


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(repo,data,roles,results,partial=False):
    t,c,_=modules(Path(repo).resolve());d=dict(np.load(data,allow_pickle=False))
    out=Path(results);design=json.loads((out/'design.json').read_text());fingerprint=design['fingerprint']
    r=pd.read_csv(roles,dtype={'point_id':str});idx={p:i for i,p in enumerate(d['point_id'])}
    splits={key:{role:np.array(sorted(idx[p] for p in z.point_id)) for role,z in g.groupby('role')}
            for key,g in r.groupby(['protocol','seed','fold'])}
    variants={v:c.transformed(d,v) for v in NEURAL};replays=[];checked=0
    cached={}
    if (out/'checkpoint_replays.csv').exists():
        cache=pd.read_csv(out/'checkpoint_replays.csv')
        if 'checkpoint_sha256' in cache:
            cached={r['run']:r for r in cache.to_dict('records')}
    paths=sorted((out/'runs').glob('*.json'))
    for path in paths:
        result=json.loads(path.read_text());assert result['fingerprint']==fingerprint
        groups=splits[(result['protocol'],result['seed'],result['fold'])]
        tr,te=groups['train'],groups['test']
        assert result['point_id']==d['point_id'][te].tolist()
        assert np.array_equal(result['actual_SOC'],d['y'][te])
        assert np.isfinite(result['pred_SOC']).all();checked+=1
        if result['model'] not in NEURAL:continue
        checkpoint_hash=sha(path.with_suffix('.pt'));record_hash=sha(path)
        old=cached.get(path.stem)
        if old is not None and old['checkpoint_sha256']==checkpoint_hash and old['record_sha256']==record_hash:
            assert old['max_abs_delta']==0
            replays.append(old);continue
        checkpoint=t.torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
        assert checkpoint['fingerprint']==fingerprint and checkpoint['parameter_count']==8449
        net=t.PatchCNN();net.load_state_dict(checkpoint['state_dict'])
        expected=t.normalization(d,tr,result['target'],'patchCNN')
        for k in expected:assert np.array_equal(expected[k],checkpoint['normalization'][k]),k
        prediction=t.neural_predict(net,checkpoint['normalization'],variants[result['model']],te,result['target'],'cpu')
        delta=float(np.max(np.abs(prediction-np.asarray(result['pred_SOC']))))
        assert delta==0,'Checkpoint forward replay differs.'
        replays.append({'run':path.stem,'max_abs_delta':delta,
                        'checkpoint_sha256':checkpoint_hash,'record_sha256':record_hash})
    if not partial:assert checked==400 and len(replays)==240
    pd.DataFrame(replays).to_csv(out/'checkpoint_replays.csv',index=False)
    status={'all_model_test_memberships_and_SOC_verified':checked,'neural_checkpoints_replayed':len(replays),
            'maximum_replay_delta':max(v['max_abs_delta'] for v in replays),
            'common_original_training_normalization_verified':True,'CNN_parameters':8449,
            'partial':partial,'scientific_verification_complete':not partial}
    (out/'verification.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--data',required=True);p.add_argument('--roles',required=True);p.add_argument('--results',required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();run(a.repo,a.data,a.roles,a.results,a.partial)
