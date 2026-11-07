"""Scientific boundary contracts; no PyTorch/GPU required for these tests."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("bruna_revision_experiments",ROOT/"scripts/bruna_revision_experiments.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def toy_manifest():
    return pd.DataFrame({"ActorID":[11,11,12,12,13,13],"file_stem":[f"v{i}" for i in range(6)],
                         "y":[0,1,2,3,4,5],"split":["train","train","val","val","test","test"],
                         "Race":["A","A","B","B","C","C"]})


class ScientificContracts(unittest.TestCase):
    def test_model_classes_match_the_executed_canonical_source(self):
        path=ROOT/'notebooks/executed/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb'
        if not path.exists():
            path=ROOT.parent/'bruna-dissertacao-colabs/notebooks/executed/08_cremad_fairness_xai_emot_definitivo_colab_pro.ipynb'
        data=path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),'3e90e89858655dfbba39a1824ddf48c0b44004f82d0c33121d453fe15e224318')
        canonical=''.join(json.loads(data)['cells'][42]['source'])
        original={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(canonical).body if isinstance(n,ast.ClassDef)}
        copied=ast.parse((ROOT/"scripts/bruna_revision_models.py").read_text(encoding="utf-8"))
        actual={n.name:ast.dump(n,include_attributes=False) for n in copied.body if isinstance(n,ast.ClassDef)}
        self.assertEqual(original,actual)

    def test_features_match_canonical_computation_including_boundaries(self):
        x=np.random.default_rng(4).normal(size=(3,100,310)).astype(np.float32)
        v=np.zeros_like(x);v[:,1:]=x[:,1:]-x[:,:-1]
        a=np.zeros_like(x);a[:,1:-1]=x[:,2:]-2*x[:,1:-1]+x[:,:-2]
        j=np.zeros_like(x);j[:,2:-2]=x[:,4:]-2*x[:,3:-1]+2*x[:,1:-3]-x[:,:-4]
        expected=np.concatenate([x,v,a,j],axis=-1)
        np.testing.assert_array_equal(m.temporal_features(x),expected)
        np.testing.assert_array_equal(m.temporal_features(x,True),x)
        self.assertFalse(np.shares_memory(m.temporal_features(x,True),x))

    def test_features_reject_incorrect_layout_and_nonfinite(self):
        with self.assertRaises(ValueError):m.temporal_features(np.zeros((1,99,310)))
        x=np.zeros((1,100,310));x[0,0,0]=np.nan
        with self.assertRaises(ValueError):m.temporal_features(x)

    def test_manifest_must_align_with_cache_and_actor_attributes(self):
        meta=toy_manifest();x=np.zeros((6,100,310));y=meta.y.to_numpy();stems=meta.file_stem.to_numpy()
        m.validate_alignment(meta,x,y,stems,expected_n=6)
        with self.assertRaises(ValueError):m.validate_alignment(meta,x,y,stems[::-1],expected_n=6)
        bad=y.copy();bad[2]=4
        with self.assertRaises(ValueError):m.validate_alignment(meta,x,bad,stems,expected_n=6)
        bad_meta=meta.copy();bad_meta.loc[1,"Race"]="different"
        with self.assertRaises(ValueError):m.validate_alignment(bad_meta,x,y,stems,expected_n=6)

    def test_fold_partitions_are_exhaustive_and_actor_disjoint(self):
        meta=toy_manifest()
        result=m.validate_fold(meta,np.array([0,1]),np.array([2,3]),np.array([4,5]))
        self.assertEqual(result,{"train_actors":1,"validation_actors":1,"test_actors":1})
        with self.assertRaises(ValueError):m.validate_fold(meta,np.array([0,2]),np.array([1,3]),np.array([4,5]))
        with self.assertRaises(ValueError):m.validate_fold(meta,np.array([0,1]),np.array([2,3]),np.array([4]))
        with self.assertRaises(ValueError):m.validate_fold(meta,np.array([0,1]),np.array([2,3]),np.array([4,4]))

    def test_macro_f1_uses_six_declared_classes_and_zero_division(self):
        y=np.arange(6)
        self.assertEqual(m.metrics(y,y)["macro_f1"],1.0)
        result=m.metrics(y,np.zeros(6,dtype=int))
        self.assertAlmostEqual(result["macro_f1"],1/21)
        self.assertAlmostEqual(result["accuracy"],1/6)
        self.assertAlmostEqual(result["balanced_accuracy"],1/6)
        with self.assertRaises(ValueError):m.metrics(y,y[:-1])

    def test_fingerprint_changes_with_seed_variant_or_fold(self):
        config={"seed":42,"variant":"EmoT_full","fold":1}
        self.assertEqual(m.fingerprint(config),m.fingerprint(dict(reversed(list(config.items())))))
        for key,value in [("seed",2026),("variant","EmoT_position"),("fold",2)]:
            changed={**config,key:value}
            self.assertNotEqual(m.fingerprint(config),m.fingerprint(changed))

    def test_resume_rejects_stale_configuration(self):
        config={"seed":42,"variant":"EmoT_full","fold":1}
        with tempfile.TemporaryDirectory() as tmp:
            original=m.lock_config(tmp,config)
            self.assertEqual(original,m.lock_config(tmp,config))
            with self.assertRaises(ValueError):m.lock_config(tmp,{**config,"seed":2026})
        with self.assertRaises(ValueError):m.require_compatible({"fingerprint":"wrong"},config)

    def test_sanity_selection_is_fixed_holdout_unique_and_deterministic(self):
        meta=pd.DataFrame({"ActorID":[11]*5+[12]*5+[13]*5+[14]*5,
                           "Race":["A"]*10+["B"]*10,"split":["train"]*5+["test"]*15})
        first=m.select_sanity_samples(meta,12,seed=8);second=m.select_sanity_samples(meta,12,seed=8)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(len(first),12);self.assertEqual(len(set(first)),12)
        self.assertTrue(meta.iloc[first].split.eq("test").all())
        self.assertLessEqual(abs(meta.iloc[first].Race.value_counts().diff().fillna(0).max()),2)

    def test_map_comparison_ties_constant_maps_and_identical_maps(self):
        a=np.array([1.,1.,2.,3.,5.]);r=m.compare_maps(a,a,3)
        self.assertAlmostEqual(r["cosine"],1.0);self.assertAlmostEqual(r["spearman"],1.0)
        self.assertEqual(r["jaccard_top_k"],1.0)
        reverse=m.compare_maps(np.arange(5),np.arange(5)[::-1],2)
        self.assertAlmostEqual(reverse["spearman"],-1.)
        self.assertEqual(reverse["jaccard_top_k"],0.)
        constant=m.compare_maps(np.zeros(5),np.zeros(5),2)
        self.assertIsNone(constant["cosine"]);self.assertIsNone(constant["spearman"])
        np.testing.assert_allclose(m._average_ranks([2,2,5,1]),[1.5,1.5,3,0])

    def test_ig_views_group_blocks_and_xy_without_scrambling_landmarks(self):
        attr=np.zeros((1,100,1240),dtype=np.float32)
        for block in range(4):attr[0,7,block*310+2*12]=block+1
        views=m._attribution_views(attr)
        self.assertEqual(views["landmark_time_absolute"].shape,(1,100,155))
        self.assertEqual(views["landmark_time_absolute"][0,7,12],10.)
        self.assertEqual(views["landmark_absolute"][0,12],10.)
        self.assertEqual(views["temporal_absolute"][0,7],10.)


if __name__=="__main__":unittest.main()
