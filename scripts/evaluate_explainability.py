"""Measure Integrated Gradients completeness, actor-level ADI and temporal deletion."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.spatial.distance import jensenshannon
import torch

from bruna_revision_experiments import _integrated_gradients, temporal_features, _seed
from bruna_revision_models import EmoTransformer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--sanity-dir", type=Path)
parser.add_argument("--run-id", default="20260809_223452_c0613d3c")
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
base = args.base_dir
work = args.output_dir
work.mkdir(parents=True, exist_ok=True)
run = base / "runs" / args.run_id / "outputs"
sanity = args.sanity_dir or base / "revision_final_20261107" / "parameter_sanity"
torch.set_num_threads(6)
device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
torch.set_float32_matmul_precision('highest')
cache = base / "cache" / "cremad_landmarks_T100_semantic.npz"
with np.load(cache, allow_pickle=False) as data:
    geometry = data['X_base']
    stems = data['file_stems'].astype(str)
assert geometry.shape == (7441, 100, 310)
assert len(set(stems)) == len(stems)
indices = {stem: i for i, stem in enumerate(stems)}
checkpoint = torch.load(run / 'models' / 'best_EmoT.pt', map_location='cpu', weights_only=False)
mu = np.asarray(checkpoint['mu'], dtype=np.float32).reshape(-1)
sigma = np.asarray(checkpoint['sigma'], dtype=np.float32).reshape(-1)
assert mu.shape == sigma.shape == (1240,) and (sigma > 0).all()
model = EmoTransformer(D=1240, n_classes=6).to(device).eval()
model.load_state_dict(checkpoint['model_state_dict'], strict=True)

def inputs(rows):
    return (temporal_features(geometry[rows]) - mu) / sigma

@torch.inference_mode()
def infer(x, logits=False):
    output = []
    for i in range(0, len(x), 32):
        scores = model(torch.from_numpy(x[i:i+32]).to(device))
        output.append((scores if logits else scores.softmax(-1)).cpu().numpy())
    return np.concatenate(output)

def save_json(name, value):
    (work / (name+'.json')).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def ig_stats(ig):
    blocks = np.abs(ig.astype(np.float64)).reshape(len(ig),100,4,310)
    mass = blocks.sum(axis=(1,3))
    return {
        'mean_clip_share': (mass / mass.sum(axis=1,keepdims=True)).mean(axis=0).tolist(),
        'mean_absolute_per_attribute_and_time': blocks.mean(axis=(0,1,3)).tolist(),
        'pooled_share': (mass.sum(0)/mass.sum()).tolist(),
    }

audit = pd.read_csv(sanity / 'sample_manifest.csv')
row = audit['sample_idx'].to_numpy()
assert np.array_equal(stems[row], audit.file_stem)
x = inputs(row)
target = audit.original_target_class.to_numpy()
original_prob = infer(x)
np.testing.assert_array_equal(original_prob.argmax(1), target)
np.testing.assert_allclose(original_prob[np.arange(len(x)),target], audit.original_predicted_probability, atol=0.001)
with np.load(sanity / 'baseline_attributions.npz', allow_pickle=False) as z:
    np.testing.assert_array_equal(z['sample_idx'], row)
    np.testing.assert_array_equal(z['target_classes'], target)
    ig = z['signed_ig']
base_ig = _integrated_gradients(model, x, target, torch, device, steps=16, batch_size=16)
cos = lambda a,b: float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
cos_original = cos(ig,base_ig)
assert cos_original > 0.995, cos_original
delta = infer(x,logits=True)[np.arange(len(x)),target] - infer(np.zeros_like(x[:1]),logits=True)[0,target]
err = np.abs(base_ig.astype(np.float64).sum(axis=(1,2)) - delta)
save_json('convergencia', {'n':len(x),'steps':16,'mean_logit_error':float(err.mean()),'median_logit_error':float(np.median(err)),'p95_logit_error':float(np.quantile(err,.95)),'max_logit_error':float(err.max()),'original_map_cosine':cos_original})
print('Completeness error calculated',flush=True)
stats = {'trained':ig_stats(ig),'randomized':{}}
for seed in (11,29,47):
    _seed(seed,torch)
    fresh = EmoTransformer(D=1240,n_classes=6)
    model.load_state_dict(fresh.state_dict(),strict=True)
    changed = _integrated_gradients(model,x,target,torch,device,steps=16,batch_size=16)
    view = np.abs(changed).reshape(len(x),100,4,155,2).sum(axis=(2,4))
    with np.load(sanity / f'seed_{seed}' / 'all_parameters.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['sample_idx'],row)
        np.testing.assert_array_equal(z['target_classes'],target)
        map_cos = cos(z['landmark_time_absolute'],view)
    assert map_cos > .995, (seed,map_cos)
    stats['randomized'][seed] = dict(ig_stats(changed),saved_map_cosine=map_cos)
    save_json('blocos_ig',stats)
    print(f'Integrated Gradients calculated for seed {seed}',flush=True)
model.load_state_dict(checkpoint['model_state_dict'],strict=True)

meta = pd.read_csv(run / 'xai' / 'xai_sample_manifest.csv')
rows = np.array([indices[s] for s in meta.file_stem])
x = inputs(rows)
probs = infer(x)
target = probs.argmax(axis=1)
base = probs.max(axis=1)
historical = pd.read_csv(run / 'xai' / 'xai_deletion_test_temporal.csv')
hist_zero = historical[historical.fraction_deleted.eq(0)].sort_values('sample_idx')
np.testing.assert_array_equal(hist_zero.sample_idx, np.arange(len(x)))
np.testing.assert_array_equal(hist_zero.Race, meta.Race)
np.testing.assert_allclose(hist_zero.masked_prob, base, atol=.001)

scores = []
for start in range(0,len(x),32):
    xb = torch.tensor(x[start:start+32],device=device,requires_grad=True)
    y = torch.as_tensor(target[start:start+32],device=device)
    logits = model(xb)
    grad, = torch.autograd.grad(logits.gather(1,y[:,None]).sum(),xb)
    attr = (grad * xb).abs().detach().cpu().numpy().reshape(len(xb),100,4,155,2)
    score = attr.sum(axis=(1,2,4),dtype=np.float64)
    scores.append(score / np.maximum(score.sum(axis=1,keepdims=True),1e-12))
landmark = np.concatenate(scores)
actors, actor_idx = np.unique(meta.ActorID.to_numpy(),return_inverse=True)
groups = np.array(['Caucasian','African American','Asian'])
actor_race = np.array([meta.Race.to_numpy()[actor_idx==i][0] for i in range(len(actors))])
assert all(len(set(meta.Race.to_numpy()[actor_idx==i])) == 1 for i in range(len(actors)))
assert set(actor_race) == set(groups)
mass = np.zeros((len(actors),155))
np.add.at(mass,actor_idx,landmark)
counts = np.bincount(actor_idx)
def adi(labels):
    averages = np.stack([mass[labels==g].sum(axis=0) / counts[labels==g].sum() for g in groups])
    return jensenshannon(np.broadcast_to(averages[0],averages[1:].shape),averages[1:],axis=1)
observed = adi(actor_race)
np.testing.assert_allclose(observed,[.0267,.0374],atol=.0002)
rng = np.random.default_rng(20261120)
permuted = np.stack([adi(rng.permutation(actor_race)) for _ in range(5000)])
np.save(work/'adi_permutacoes.npy',permuted)
p = ((permuted >= observed).sum(axis=0)+1)/5001
max_p = ((permuted.max(axis=1)[:,None]>=observed).sum(axis=0)+1)/5001
save_json('adi', {'actors':len(actors),'clips':len(x),'n_actors_by_group':dict(zip(groups.tolist(),[int((actor_race==g).sum()) for g in groups])),'groups':groups[1:].tolist(),'observed':observed.tolist(),'p':p.tolist(),'p_max_statistic':max_p.tolist()})
print('Actor-level ADI permutation completed',flush=True)

fractions = np.sort(historical.fraction_deleted.unique())
n_draws = 5
orders = np.argsort(rng.random((n_draws,len(x),100)),axis=2)
rng_masked = []
for frac in fractions:
    k = round(float(frac)*100)
    if k == 0:
        mean_prob = base
    elif k == 100:
        zeros = np.zeros_like(x[:1])
        mean_prob = infer(zeros)[0,target]
    else:
        estimates = []
        for order in orders:
            masked = x.copy()
            masked[np.arange(len(x))[:,None],order[:,:k],:] = 0
            estimates.append(infer(masked)[np.arange(len(x)),target])
        mean_prob = np.mean(estimates,axis=0)
    rng_masked.append(mean_prob)
    print(f'Random temporal deletion: {frac:.0%}',flush=True)
random = np.stack(rng_masked,axis=1)
order_hist = historical.pivot(index='sample_idx',columns='fraction_deleted',values='masked_prob').to_numpy()
np.testing.assert_allclose(random[:,[0,-1]],order_hist[:,[0,-1]],atol=.001)
for i, frac in enumerate(fractions):
    historical.loc[historical.fraction_deleted.eq(frac),'random_masked_prob'] = random[historical.loc[historical.fraction_deleted.eq(frac),'sample_idx'],i]
historical.to_csv(work/'delecao_contraste.csv',index=False)
save_json('delecao', {'clips':len(x),'draws_per_clip':n_draws,'fractions':fractions.tolist(),'attention_curve':order_hist.mean(axis=0).tolist(),'random_curve':random.mean(axis=0).tolist(),'attention_auc':float(trapezoid(order_hist.mean(axis=0),fractions)),'random_auc':float(trapezoid(random.mean(axis=0),fractions))})
print('Explainability evaluation completed',flush=True)
