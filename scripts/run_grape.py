#!/usr/bin/env python3
"""Build inputs, train GRAPE-UPS alone, or evaluate archived GRAPE-UPS weights."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ups_ai import models as m
from ups_ai.common import load_config,set_seed,robust_center_scale,artifact_dir
from ups_ai.features import build_features
from ups_ai.evaluation import threshold_from_validation,binary_metrics,repeated_event_ewma
from sklearn.metrics import average_precision_score

def execute(config_path,mode,output,epochs=None,dataset=None,model_path=None):
    cfg=load_config(config_path)
    if dataset is not None:
        entry=next((x for x in json.loads((ROOT/'data/datasets.json').read_text()) if x['dataset']==dataset),None)
        if entry is None or 'seed' not in entry:raise ValueError('Choose a primary/development dataset from data/datasets.json.')
        cfg.update(dataset_name=dataset,data_path=entry['path'],seed=entry['seed'],difficulty=entry['difficulty'],packet_loss_rate=entry['packet_loss_rate'],upload_latency_sigma_s=entry['upload_latency_sigma_s'])
        cfg['model']['random_state']=entry['seed']
    if 'dataset_name' not in cfg:raise ValueError('Use --dataset with configs/train.json.')
    name=cfg['dataset_name'];device=torch.device('cpu');torch.set_num_threads(1)
    model_dir=ROOT/Path(model_path or cfg.get('model_path','models/example')) if mode=='evaluate' else output/'models'
    if mode=='evaluate' and (model_dir/'model.json').exists():
        expected=json.loads((model_dir/'model.json').read_text()).get('dataset')
        if expected and expected!=name:raise ValueError('The model belongs to '+expected+'; train this dataset or supply its matching --model-dir.')
    if not all((artifact_dir(cfg)/f).exists() for f in ['event_features.csv','event_sequences.npz','feature_manifest.json']):build_features(cfg,force=True)
    data=m._prepare_data(cfg)
    if mode=='train':
        model_dir.mkdir(parents=True,exist_ok=True);set_seed(int(cfg['seed']))
        if epochs is not None:cfg['model']['epochs']=epochs
        rel,_=m._train_hybrid_autoencoder(data.seq_all[data.train_mask],data.x_all[data.train_mask],cfg,device)
        absolute,_=m._train_hybrid_autoencoder(data.seq_abs_all[data.train_mask],data.x_abs_all[data.train_mask],cfg,device)
        torch.save(rel.state_dict(),model_dir/'grape_hybrid_autoencoder.pt')
        torch.save(absolute.state_dict(),model_dir/'grape_no_group_autoencoder.pt')
    else:
        rel=m.HybridAutoencoder(data.seq_all.shape[1],data.seq_all.shape[2],data.x_all.shape[1],int(cfg['model']['latent_dim']))
        absolute=m.HybridAutoencoder(data.seq_abs_all.shape[1],data.seq_abs_all.shape[2],data.x_abs_all.shape[1],int(cfg['model']['latent_dim']))
        rel.load_state_dict(torch.load(model_dir/'grape_hybrid_autoencoder.pt',map_location='cpu',weights_only=True))
        absolute.load_state_dict(torch.load(model_dir/'grape_no_group_autoencoder.pt',map_location='cpu',weights_only=True))
    components=m._score_components(data,rel,absolute,device)
    weights,_=m._load_fusion_weights(cfg);fusion=m._fusion_matrix(components)@weights;peer=components['group_score_z']
    val=data.validation_mask;yv=data.features.loc[val,'label_anomaly'].to_numpy(int)
    if len(np.unique(yv))<2:raise ValueError('Validation requires both classes for alert-head selection.')
    head='fusion' if average_precision_score(yv,fusion[val])>average_precision_score(yv,peer[val]) else 'peer'
    selected=fusion if head=='fusion' else peer;threshold=threshold_from_validation(yv,selected[val])
    # Fixed alpha is the reported extension; selection uses validation, never held-out labels.
    accumulated=repeated_event_ewma(data.features.reset_index(drop=True),fusion,float(cfg.get('accumulator_alpha',0.8)))
    accumulated_threshold=threshold_from_validation(yv,accumulated[val])
    out=data.features[[c for c in ['string_id','event_id','event_index','battery_id','split','label_anomaly','label_battery_weak','fault_severity','active_fault_type'] if c in data.features]].copy()
    for key,value in components.items():out['component_'+key]=value
    out['raw_g']=components['group_score'];out['peer_score']=peer;out['fusion_score']=fusion
    out['selected_alert_score']=selected;out['selected_alert']=selected>=threshold;out['accumulated_fusion']=accumulated
    output.mkdir(parents=True,exist_ok=True);out.to_csv(output/'scores.csv',index=False)
    metrics={}
    for split,mask in [('test_seen',data.test_seen_mask),('test_external',data.test_external_mask)]:
        y=data.features.loc[mask,'label_anomaly'].to_numpy(int)
        if not len(y) or len(np.unique(y))<2:continue
        metrics[split]=binary_metrics(y,selected[mask],threshold)
    settings={'dataset':name,'ranking_score':'peer_score','alert_head':head,'threshold':threshold,'accumulator_alpha':float(cfg.get('accumulator_alpha',0.8)),'accumulator_threshold':accumulated_threshold,'fusion_component_order':m.FUSION_COMPONENT_ORDER,'fusion_weights':weights.tolist(),'component_normalization':{k:dict(zip(['median','scale'],robust_center_scale(v[data.train_mask]))) for k,v in components.items() if not k.endswith('_z')}}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2));(output/'settings.json').write_text(json.dumps(settings,indent=2))
    if mode=='train':(model_dir/'model.json').write_text(json.dumps(settings,indent=2))
    return data,settings,out

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--mode',choices=['evaluate','train'],default='evaluate');p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int);p.add_argument('--dataset',help='Dataset key from data/datasets.json');p.add_argument('--model-dir',type=Path,help='Matching model directory for evaluation');a=p.parse_args()
    execute(a.config,a.mode,a.output,a.epochs,a.dataset,a.model_dir)
if __name__=='__main__':main()
