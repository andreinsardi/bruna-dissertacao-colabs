"""Frozen CREMA-D -> RAVDESS visual evaluation. Nothing executes on import."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlparse
import zipfile

import numpy as np
import pandas as pd

from bruna_revision_experiments import (
    CLASSES, CANONICAL_RUN, temporal_features, metrics, hash_file,
    atomic_csv, atomic_json, atomic_npz, lock_config, fingerprint,
)
from bruna_revision_statistics import confusion, metrics_from_confusion

PROTOCOL = 'bruna-ravdess-frozen-external-20261107-v2-env-recovery'
LABEL_MAP = {5: 0, 7: 1, 6: 2, 3: 3, 1: 4, 4: 5}


def parse_ravdess(filename):
    stem = Path(filename).stem
    match = re.fullmatch(r'(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', stem)
    if not match:
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = map(int, match.groups())
    if modality not in (1,2,3) or channel not in (1,2) or emotion not in range(1,9):
        raise ValueError('Invalid RAVDESS modality/channel/emotion: '+stem)
    if actor not in range(1,25) or statement not in (1,2) or repetition not in (1,2) or intensity not in (1,2):
        raise ValueError('Invalid RAVDESS actor or stimulus metadata: '+stem)
    if emotion == 1 and intensity != 1:
        raise ValueError('Neutral has no strong intensity: '+stem)
    if modality != 2 or channel != 1 or emotion not in LABEL_MAP:
        return None
    return {'dataset': 'RAVDESS', 'file_stem': stem, 'ActorID': actor,
            'identity_namespace': f'RAVDESS:{actor:02d}', 'y': LABEL_MAP[emotion],
            'label_name': CLASSES[LABEL_MAP[emotion]], 'emotion_code': emotion,
            'intensity': intensity, 'statement': statement, 'repetition': repetition,
            'modality': modality, 'vocal_channel': channel}


def expected_stems():
    return {f'02-01-{e:02d}-{i:02d}-{s:02d}-{r:02d}-{a:02d}'
            for a in range(1,25) for e in LABEL_MAP
            for i in ((1,) if e == 1 else (1,2)) for s in (1,2) for r in (1,2)}


def validate_manifest(manifest):
    if 'file_stem' not in manifest or manifest.file_stem.duplicated().any():
        raise ValueError('Missing or duplicated external stimuli.')
    if set(manifest.file_stem) != expected_stems() or len(manifest) != 1056:
        raise ValueError('The complete 24-actor/1,056-stimulus external set is required.')
    for row in manifest.to_dict('records'):
        parsed = parse_ravdess(row['file_stem']+'.mp4')
        if parsed is None or any(row.get(k) != parsed[k] for k in ('ActorID','y','label_name')):
            raise ValueError('Manifest metadata differs from official filename.')


def validate_checkpoint_normalization(payload):
    if list(payload.get('class_names', [])) != CLASSES:
        raise ValueError('Checkpoint class order differs from CREMA-D.')
    mu, sigma = [np.asarray(payload[k], dtype=np.float32).reshape(-1) for k in ('mu','sigma')]
    if mu.shape != (1240,) or sigma.shape != (1240,) or not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('Invalid saved CREMA-D training normalization.')
    return mu, sigma


def validate_external_alignment(manifest, x, cache, canonical_ids):
    if manifest.file_stem.duplicated().any() or not len(manifest):
        raise ValueError('Duplicated or empty external manifest.')
    for row in manifest.to_dict('records'):
        parsed=parse_ravdess(row['file_stem']+'.mp4')
        if parsed is None or any(row.get(k)!=parsed[k] for k in ('ActorID','identity_namespace','y','label_name')):
            raise ValueError('External metadata differs from stimulus identity.')
    if x.shape!=(len(manifest),100,310) or not np.isfinite(x).all():
        raise ValueError('Invalid external input tensor.')
    if not np.array_equal(manifest.file_stem.to_numpy(str),cache['file_stems'].astype(str)):
        raise ValueError('External manifest/cache stimulus order mismatch.')
    if not np.array_equal(manifest.y.to_numpy(int),cache['y']):
        raise ValueError('External manifest/cache label mismatch.')
    if not np.array_equal(x,cache['X_base']):
        raise ValueError('Passed external tensor differs from the persisted cache.')
    if len(canonical_ids)!=155 or len(np.unique(canonical_ids))!=155 or not np.array_equal(cache['selected_landmark_ids'],canonical_ids):
        raise ValueError('External/canonical landmark IDs or order differ.')


def verify_canonical_extraction(base_dir, output_dir, atol=1e-4, mean_atol=1e-6):
    """Check six original clips before downloading an external dataset."""
    import importlib.metadata
    import bruna_ravdess_canonical_extractor as extractor
    from bruna_revision_experiments import _torch_load
    from bruna_revision_models import EmoTransformer, TemporalCNN1D
    import torch
    base,root=Path(base_dir),Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    if importlib.metadata.version('mediapipe')!='0.10.35' or extractor.cv2.__version__!='4.13.0':
        raise RuntimeError('Restart after installing MediaPipe 0.10.35 and OpenCV headless 4.13.0.92.')
    task=base/'cache/face_landmarker.task'
    if not task.is_file():raise FileNotFoundError('Preserved canonical task model missing.')
    extractor.FACE_LANDMARKER_MODEL_PATH=task
    source=base/'runs'/CANONICAL_RUN/'outputs'
    manifest=pd.read_csv(source/'tables/manifest_valid_with_splits.csv')
    if len(manifest)!=7441 or manifest.file_stem.duplicated().any():
        raise ValueError('Canonical manifest changed.')
    rng=np.random.default_rng(20261107);selected=[];used=set()
    for label in range(6):
        candidates=manifest.index[manifest.y.eq(label) & ~manifest.ActorID.isin(used)].to_numpy()
        chosen=int(rng.choice(candidates));selected.append(chosen);used.add(manifest.loc[chosen,'ActorID'])
    with np.load(base/'cache/cremad_landmarks_T100_semantic.npz',allow_pickle=False) as cache:
        if not np.array_equal(cache['file_stems'].astype(str),manifest.file_stem.to_numpy(str)) or not np.array_equal(cache['y'],manifest.y.to_numpy(int)):
            raise ValueError('Canonical tensor order/labels differ from its manifest.')
        if not np.array_equal(cache['selected_landmark_ids'],extractor.SELECTED_LANDMARK_IDS):
            raise ValueError('Canonical landmark selection differs from the extractor.')
        original=cache['X_base'][selected].copy()
    # Reject checkpoint/schema/runtime problems before any external download.
    for name in ('EmoT','TemporalCNN1D'):
        payload=_torch_load(source/f'models/best_{name}.pt',torch)
        validate_checkpoint_normalization(payload)
        model=EmoTransformer(D=1240,n_classes=6) if name=='EmoT' else TemporalCNN1D(T=100,D=1240,n_classes=6)
        model.load_state_dict(payload['model_state_dict'],strict=True)
        del model,payload
    records=[]
    with extractor.create_face_landmarker() as detector:
        for reference,index in zip(original,selected):
            row=manifest.loc[index].to_dict()
            if not Path(row['video_path']).is_file():raise FileNotFoundError(row['video_path'])
            sequence,report=extractor.process_one_video(row,detector)
            if sequence is None:
                records.append({'file_stem':row['file_stem'],'status':report['status'],'match':False})
                continue
            difference=np.abs(reference-sequence)
            maximum,mean=float(difference.max()),float(difference.mean())
            records.append({'file_stem':row['file_stem'],'ActorID':int(row['ActorID']),'y':int(row['y']),
                            'status':report['status'],'max_abs_difference':maximum,'mean_abs_difference':mean,
                            'match':maximum<=atol and mean<=mean_atol,'video_sha256':hash_file(row['video_path'])})
    frame=pd.DataFrame(records);atomic_csv(root/'canonical_extractor_preflight.csv',frame)
    evidence={'all_six_passed':bool(frame.match.all()),'selection_seed':20261107,'n_samples':6,
              'max_atol':atol,'mean_atol':mean_atol,'mediapipe':'0.10.35','task_sha256':hash_file(task),
              'canonical_run':CANONICAL_RUN,'extractor_sha256':hash_file(extractor.__file__),'opencv':extractor.cv2.__version__,
              'manifest_sha256':hash_file(source/'tables/manifest_valid_with_splits.csv'),
              'samples':records,'scope':'numerical extraction consistency only; not a new performance estimate'}
    atomic_json(root/'canonical_extractor_preflight.json',evidence)
    if not evidence['all_six_passed']:
        raise RuntimeError('Re-extraction differs from preserved CREMA-D tensors; investigate before external evaluation.')
    return evidence


def conditional_actor_bootstrap(y, predictions, actors, n_boot=5000, seed=20261107):
    y, actors = np.asarray(y, dtype=int), np.asarray(actors)
    if y.ndim != 1 or actors.shape != y.shape or not len(y) or not predictions or n_boot < 2:
        raise ValueError('Invalid actor bootstrap input.')
    identities = np.unique(actors)
    if len(identities) < 2:
        raise ValueError('At least two external identities are required for an interval.')
    matrices = {}
    for model, predicted in predictions.items():
        predicted = np.asarray(predicted, dtype=int)
        if predicted.shape != y.shape:
            raise ValueError('External predictions are not paired.')
        matrices[model] = np.stack([confusion(y[actors==a], predicted[actors==a]) for a in identities])
    rng = np.random.default_rng(seed)
    # One shared multiplicity matrix preserves model pairing at actor level.
    weights = rng.multinomial(len(identities), np.ones(len(identities))/len(identities), size=n_boot)
    distribution, point = {}, {}
    for model, cm in matrices.items():
        distribution[model] = metrics_from_confusion(np.einsum('ba,aij->bij',weights,cm))
        point[model] = metrics_from_confusion(cm.sum(0))
    records = []
    names = list(predictions)
    contrasts = [(m,None) for m in names] + [(names[i],names[j]) for i in range(len(names)) for j in range(i+1,len(names))]
    for left, right in contrasts:
        for metric in ('accuracy','macro_f1'):
            values = distribution[left][metric] - (distribution[right][metric] if right else 0)
            estimate = point[left][metric] - (point[right][metric] if right else 0)
            lo,hi = np.quantile(values,[.025,.975])
            records.append({'contrast': f'{left} minus {right}' if right else left, 'metric':metric,
                            'estimate':float(estimate),'ci_low':float(lo),'ci_high':float(hi),
                            'n_actors':len(identities),'n_videos':len(y),'n_boot':n_boot,'seed':seed,
                            'scope':'conditional on frozen fitted models and successfully extracted clips'})
    return pd.DataFrame(records)


def _download_archive(record, destination):
    import requests
    parsed = urlparse(record['url'])
    if parsed.scheme != 'https' or parsed.hostname != 'zenodo.org':
        raise ValueError('Expected the official HTTPS Zenodo archive endpoint.')
    algorithm, expected = record['checksum'].split(':',1)
    if algorithm != 'md5' or not re.fullmatch('[0-9a-f]{32}',expected):
        raise ValueError('Unexpected official archive checksum format.')
    if destination.is_file() and destination.stat().st_size == int(record['size']):
        digest=hashlib.md5()
        with destination.open('rb') as handle:
            for chunk in iter(lambda:handle.read(2**20),b''):digest.update(chunk)
        if digest.hexdigest() == expected:
            return
    partial=destination.with_suffix('.download.partial')
    error=None
    for attempt in range(3):
        try:
            digest=hashlib.md5(); size=0
            with requests.get(record['url'],stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with partial.open('wb') as handle:
                    for chunk in response.iter_content(2**20):
                        if chunk:handle.write(chunk);digest.update(chunk);size+=len(chunk)
            if size != int(record['size']) or digest.hexdigest() != expected:
                raise ValueError('Official RAVDESS archive checksum/size mismatch.')
            partial.replace(destination)
            return
        except Exception as exc:
            error=exc
            print(f'Archive attempt {attempt+1}/3 failed: {type(exc).__name__}',flush=True)
    raise RuntimeError('Unable to obtain verified RAVDESS archive.') from error


def prepare_ravdess_videos(output_dir, archive_manifest, scratch_dir='/content/bruna_ravdess_archives'):
    root=Path(output_dir); raw=root/'raw'; raw.mkdir(parents=True,exist_ok=True)
    scratch=Path(scratch_dir);scratch.mkdir(parents=True,exist_ok=True)
    entries=archive_manifest['files']
    expected_names={f'Video_Speech_Actor_{a:02d}.zip' for a in range(1,25)}
    if len(entries)!=24 or {r['name'] for r in entries} != expected_names:
        raise ValueError('Official manifest must contain exactly 24 speech archives.')
    config={'version':PROTOCOL,'archive_manifest':archive_manifest,'selection':'VO02/speech01/six common classes; no calm-neutral merge'}
    fp=lock_config(raw,config)
    rows=[]; excluded=[]
    for entry in sorted(entries,key=lambda r:r['name']):
        actor=int(entry['name'][-6:-4])
        marker=raw/f'actor_{actor:02d}.json'
        if marker.is_file():
            existing=json.loads(marker.read_text())
            if existing.get('fingerprint')!=fp:raise ValueError('Incompatible raw-video resume.')
            for record in existing['videos']:
                path=Path(record['video_path'])
                if not path.is_file() or hash_file(path)!=record['video_sha256']:
                    raise ValueError('Persisted raw video changed; do not reuse this extraction.')
            rows.extend(existing['videos']);excluded.extend(existing['excluded'])
            print(f'RAVDESS actor {actor:02d}: verified resume',flush=True)
            continue
        if shutil.disk_usage(scratch).free < int(entry['size'])*2:
            raise RuntimeError('Insufficient temporary disk for the next actor archive.')
        archive=scratch/entry['name']
        print('Downloading official archive:',entry['name'],flush=True)
        _download_archive(entry,archive)
        actor_rows=[];actor_excluded=[];seen=set()
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                if item.is_dir() or not item.filename.lower().endswith('.mp4'):continue
                name=Path(item.filename).name
                parsed=parse_ravdess(name)
                if parsed is None:
                    actor_excluded.append({'archive':entry['name'],'filename':item.filename,'reason':'outside predeclared modality/channel/classes'})
                    continue
                if parsed['ActorID'] != actor or parsed['file_stem'] in seen:
                    raise ValueError('Actor mismatch or duplicated stimulus inside archive.')
                seen.add(parsed['file_stem'])
                # Never use an archive-provided path as a destination.
                target=raw/f'Actor_{actor:02d}'/(parsed['file_stem']+'.mp4')
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix('.mp4.partial')
                with bundle.open(item) as source, temporary.open('wb') as dest:
                    shutil.copyfileobj(source,dest,2**20)
                temporary.replace(target)
                parsed.update(video_path=str(target),video_sha256=hash_file(target),archive_name=entry['name'])
                actor_rows.append(parsed)
        if seen != {s for s in expected_stems() if s.endswith(f'-{actor:02d}')}:
            raise ValueError('Actor archive does not contain all 44 selected stimuli.')
        atomic_json(marker,{'fingerprint':fp,'videos':actor_rows,'excluded':actor_excluded})
        rows.extend(actor_rows);excluded.extend(actor_excluded)
        # Only this scratch archive is removed, after verified video persistence.
        archive.unlink()
        print(f'RAVDESS actor {actor:02d}: {len(actor_rows)} eligible clips persisted',flush=True)
    frame=pd.DataFrame(rows).sort_values('file_stem').reset_index(drop=True)
    validate_manifest(frame)
    atomic_csv(root/'manifest_nominal.csv',frame)
    atomic_csv(root/'excluded_stimuli.csv',pd.DataFrame(excluded))
    return frame


def extract_ravdess(base_dir, output_dir, manifest):
    import importlib.metadata
    from bruna_ravdess_canonical_extractor import process_one_video, create_face_landmarker
    import bruna_ravdess_canonical_extractor as extractor
    base,root=Path(base_dir),Path(output_dir)
    validate_manifest(manifest)
    version=importlib.metadata.version('mediapipe')
    if version!='0.10.35' or extractor.cv2.__version__!='4.13.0':
        raise RuntimeError('Extraction requires the pinned MediaPipe 0.10.35/OpenCV 4.13.0 configuration and a passing preflight.')
    task=base/'cache/face_landmarker.task'
    if not task.is_file():raise FileNotFoundError('The preserved canonical FaceLandmarker model is required; no latest-model download is substituted.')
    with np.load(base/'cache/cremad_landmarks_T100_semantic.npz',allow_pickle=False) as cache:
        ids=cache['selected_landmark_ids'].astype(int)
    if not np.array_equal(ids,extractor.SELECTED_LANDMARK_IDS):
        raise ValueError('The 155 selected landmark IDs/order differ from CREMA-D.')
    extractor.FACE_LANDMARKER_MODEL_PATH=task
    config={'protocol':PROTOCOL,'mediapipe':version,'opencv':extractor.cv2.__version__,'backend':'tasks/IMAGE/CPU',
            'canonical_task_sha256':hash_file(task),'extractor_source_sha256':hash_file(extractor.__file__),
            'manifest_records':manifest[['file_stem','video_sha256']].to_dict('records'),
            'T':100,'minimum_detection_rate':.60,'selected_landmark_ids':ids.tolist()}
    shards=root/'extraction_shards';fp=lock_config(shards,config)
    reports=[];valid_stems=[];sequences=[]
    with create_face_landmarker() as detector:
        for index,row in enumerate(manifest.to_dict('records')):
            path=shards/(row['file_stem']+'.npz');meta=path.with_suffix('.json')
            if meta.is_file():
                saved=json.loads(meta.read_text())
                if saved.get('fingerprint')!=fp or hash_file(path)!=saved['sha256']:
                    raise ValueError('Invalid external-extraction resume.')
                report=saved['report']
                with np.load(path,allow_pickle=False) as z:
                    sequence=z['sequence']
                if report['status']=='ok' and (sequence.shape!=(100,310) or not np.isfinite(sequence).all()):
                    raise ValueError('Invalid persisted external landmarks.')
            else:
                if hash_file(row['video_path']) != row['video_sha256']:
                    raise ValueError('Raw video changed after the frozen manifest.')
                sequence,report=process_one_video(row,detector)
                # An unexpected extractor error aborts instead of becoming a silent exclusion.
                if sequence is not None and (sequence.shape!=(100,310) or not np.isfinite(sequence).all()):
                    raise ValueError('Unexpected landmark shape or nonfinite values.')
                atomic_npz(path,sequence=sequence if sequence is not None else np.empty((0,310),np.float32))
                atomic_json(meta,{'fingerprint':fp,'report':report,'sha256':hash_file(path)})
            reports.append({**row,**report})
            if report['status']=='ok':
                sequences.append(sequence);valid_stems.append(row['file_stem'])
            if (index+1)%10==0:print(f'RAVDESS extraction: {index+1}/1056; valid={len(valid_stems)}',flush=True)
    if not sequences:raise RuntimeError('No external clip met the canonical detection threshold.')
    report_frame=pd.DataFrame(reports)
    atomic_csv(root/'processing_report.csv',report_frame)
    valid=report_frame[report_frame.status.eq('ok')].reset_index(drop=True)
    if valid.file_stem.tolist()!=valid_stems:raise ValueError('External tensor/manifest alignment failed.')
    atomic_csv(root/'manifest_valid.csv',valid)
    coverage=report_frame.groupby(['ActorID','label_name']).agg(n_nominal=('status','size'),n_valid=('status',lambda s:int(s.eq('ok').sum()))).reset_index()
    atomic_csv(root/'coverage_actor_emotion.csv',coverage)
    x=np.stack(sequences).astype(np.float32)
    atomic_npz(root/'ravdess_landmarks_T100_semantic.npz',X_base=x,y=valid.y.to_numpy(int),file_stems=np.asarray(valid_stems),selected_landmark_ids=ids)
    atomic_json(root/'extraction_completed.json',{'fingerprint':fp,'n_nominal':1056,'n_processed':len(reports),'n_valid':len(valid),'n_valid_actors':int(valid.ActorID.nunique()),'cache_sha256':hash_file(root/'ravdess_landmarks_T100_semantic.npz'),'valid_manifest_sha256':hash_file(root/'manifest_valid.csv'),'processing_report_sha256':hash_file(root/'processing_report.csv')})
    return valid,x


def evaluate_external(base_dir, output_dir, valid_manifest, x_base, n_boot=5000):
    import torch
    from bruna_revision_models import EmoTransformer, TemporalCNN1D
    from bruna_revision_experiments import _torch_load, _predict, _environment
    base,root=Path(base_dir),Path(output_dir)
    with np.load(base/'cache/cremad_landmarks_T100_semantic.npz',allow_pickle=False) as canonical:
        canonical_ids=canonical['selected_landmark_ids']
    with np.load(root/'ravdess_landmarks_T100_semantic.npz',allow_pickle=False) as cache:
        validate_external_alignment(valid_manifest,x_base,cache,canonical_ids)
    persisted=pd.read_csv(root/'manifest_valid.csv')
    if not persisted[['file_stem','ActorID','y']].equals(valid_manifest[['file_stem','ActorID','y']].reset_index(drop=True)):
        raise ValueError('Passed external manifest differs from persisted manifest.')
    report=pd.read_csv(root/'processing_report.csv');validate_manifest(report)
    accepted=report.loc[report.status.eq('ok'),'file_stem'].tolist()
    if accepted!=valid_manifest.file_stem.tolist():raise ValueError('Valid subset differs from the full processing report.')
    extraction=json.loads((root/'extraction_completed.json').read_text())
    for key,path in [('cache_sha256',root/'ravdess_landmarks_T100_semantic.npz'),('valid_manifest_sha256',root/'manifest_valid.csv'),('processing_report_sha256',root/'processing_report.csv')]:
        if extraction[key]!=hash_file(path):raise ValueError('External extraction artifact hash mismatch.')
    run=base/'runs'/CANONICAL_RUN/'outputs'
    checkpoints={'EmoT':run/'models/best_EmoT.pt','TemporalCNN1D':run/'models/best_TemporalCNN1D.pt'}
    config={'protocol':PROTOCOL,'training_dataset':'CREMA-D only','target_dataset':'RAVDESS',
            'canonical_run':CANONICAL_RUN,'checkpoint_sha256':{k:hash_file(p) for k,p in checkpoints.items()},
            'cache_sha256':hash_file(root/'ravdess_landmarks_T100_semantic.npz'),
            'valid_manifest_sha256':hash_file(root/'manifest_valid.csv'),'processing_report_sha256':hash_file(root/'processing_report.csv'),
            'extraction_completion_sha256':hash_file(root/'extraction_completed.json'),
            'source_sha256':hash_file(__file__),'model_source_sha256':hash_file(Path(__file__).with_name('bruna_revision_models.py')),
            'environment':_environment(torch),'normalization':'saved CREMA-D training mean/std; no target fitting',
            'selection':'all predeclared successfully extracted clips; no RAVDESS tuning','n_boot':int(n_boot)}
    results=root/'evaluation';fp=lock_config(results,config)
    predictions={};summary=[];class_rows=[];actor_rows=[]
    x=temporal_features(x_base);device='cuda' if torch.cuda.is_available() else 'cpu'
    y=valid_manifest.y.to_numpy(int)
    for name,path in checkpoints.items():
        payload=_torch_load(path,torch);mu,sigma=validate_checkpoint_normalization(payload)
        standardized=(x-mu)/sigma
        if not np.isfinite(standardized).all():raise ValueError('Invalid fixed-normalization inputs.')
        model=(EmoTransformer(D=1240,n_classes=6) if name=='EmoT' else TemporalCNN1D(T=100,D=1240,n_classes=6)).to(device)
        model.load_state_dict(payload['model_state_dict'],strict=True)
        pred,prob,logits=_predict(model,standardized,torch,device,batch_size=32)
        predictions[name]=pred
        frame=valid_manifest[['file_stem','ActorID','identity_namespace','y','label_name']].copy()
        frame['model']=name;frame['y_pred']=pred
        for i,label in enumerate(CLASSES):frame['prob_'+label]=prob[:,i]
        atomic_csv(results/(name+'_predictions.csv'),frame)
        atomic_npz(results/(name+'_logits.npz'),logits=logits,probabilities=prob,y_true=y,y_pred=pred,file_stems=valid_manifest.file_stem.to_numpy(str))
        summary.append({'model':name,'n_videos':len(y),'n_actors':int(valid_manifest.ActorID.nunique()),**metrics(y,pred)})
        cm=confusion(y,pred);atomic_csv(results/(name+'_confusion.csv'),pd.DataFrame(cm,index=CLASSES,columns=CLASSES).reset_index(names='true_class'))
        for i,label in enumerate(CLASSES):
            support=int(cm[i].sum())
            class_rows.append({'model':name,'emotion':label,'n_videos':support,'n_actors':int(valid_manifest.loc[valid_manifest.y.eq(i),'ActorID'].nunique()),'true_positive':int(cm[i,i]),'recall':float(cm[i,i]/support) if support else None})
        for actor,part in frame.groupby('ActorID'):
            actor_rows.append({'model':name,'ActorID':int(actor),'n_videos':len(part),**metrics(part.y,part.y_pred)})
        del model,payload,standardized;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        print('External inference complete:',name,summary[-1],flush=True)
    atomic_csv(results/'metrics.csv',pd.DataFrame(summary))
    atomic_csv(results/'recall_by_emotion.csv',pd.DataFrame(class_rows))
    atomic_csv(results/'metrics_per_actor.csv',pd.DataFrame(actor_rows))
    intervals=conditional_actor_bootstrap(y,predictions,valid_manifest.ActorID.to_numpy(),n_boot=n_boot)
    atomic_csv(results/'actor_cluster_intervals.csv',intervals)
    extraction=json.loads((root/'extraction_completed.json').read_text())
    if extraction['n_nominal']!=1056 or extraction['n_processed']!=1056:
        raise RuntimeError('Incomplete external extraction; completion cannot be declared.')
    artifact_records=[{'path':str(p.relative_to(results)),'bytes':p.stat().st_size,'sha256':hash_file(p)} for p in sorted(results.rglob('*')) if p.is_file() and p.name!='completed_external_evaluation.json']
    completion={'status':'completed_external_evaluation','fingerprint':fp,'scope':'frozen original CREMA-D checkpoints; post-development cross-dataset evaluation',
                'n_nominal':1056,'n_valid':len(y),'n_failed_detection':1056-len(y),'n_valid_actors':int(valid_manifest.ActorID.nunique()),
                'gpu_used':device=='cuda','no_finetuning':True,'n_boot':n_boot,'artifacts':artifact_records,
                'limitations':['RAVDESS was used earlier in project development','Intervals conditional on frozen fits and valid detections','No racial/ethnic/age attributes inferred','Different from within-RAVDESS training/testing protocols']}
    atomic_json(results/'completed_external_evaluation.json',completion)
    return completion
