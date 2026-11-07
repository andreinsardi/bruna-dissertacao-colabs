import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('bruna_statistics',Path(__file__).resolve().parents[1]/'scripts/bruna_revision_statistics.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class ActorStatisticsTests(unittest.TestCase):
    def test_six_class_macro_f1_and_undefined_classes(self):
        cm=s.confusion([0,0,1,1],[0,1,1,1])
        expected=((2/3)+(4/5))/6
        self.assertAlmostEqual(float(s.metrics_from_confusion(cm)['macro_f1']),expected)
        self.assertEqual(float(s.metrics_from_confusion(np.zeros((6,6)))['macro_f1']),0.)

    def test_cluster_aggregation_preserves_whole_actor_contribution(self):
        frame=pd.DataFrame({'ActorID':[10,10,20],'y_true':[0,0,1],'y_pred':[0,1,1]})
        cm=s.actor_confusions(frame,np.array([10,20]))
        self.assertEqual(cm[0].sum(),2);self.assertEqual(cm[1].sum(),1)
        # Drawing actor 10 twice carries both of that actor's videos twice.
        drawn=2*cm[0]
        self.assertEqual(drawn.sum(),4);self.assertEqual(drawn[0,1],2)

    def test_paired_identity_has_exact_zero_delta_and_is_deterministic(self):
        cm=np.zeros((3,6,6));cm[0,0,0]=3;cm[1,1,2]=4;cm[2,2,2]=2
        a, draws=s.cluster_bootstrap_pair(cm,cm,n_boot=100,seed=44)
        b, draws2=s.cluster_bootstrap_pair(cm,cm,n_boot=100,seed=44)
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(draws,draws2)
        self.assertTrue((draws.to_numpy()==0).all())

    def test_equal_actor_weight_changes_estimand(self):
        cm=np.zeros((2,6,6));cm[0,0,0]=100;cm[1,1,0]=1
        video,_=s.cluster_bootstrap_pair(cm,cm,20,4,False)
        person,_=s.cluster_bootstrap_pair(cm,cm,20,4,True)
        va=video.query("target=='EmoT' and metric=='accuracy'").observed.iloc[0]
        pa=person.query("target=='EmoT' and metric=='accuracy'").observed.iloc[0]
        self.assertAlmostEqual(va,100/101);self.assertEqual(pa,.5)

    def test_pair_rejects_different_actor_support(self):
        a=np.zeros((2,6,6));a[:,0,0]=1;b=a.copy();b[0,0,0]=2
        with self.assertRaisesRegex(ValueError,'per-actor support'):
            s.cluster_bootstrap_pair(a,b)

    def test_permutation_moves_categories_as_actor_units_within_fold(self):
        cats=np.array(['A','A','B','C','C','D']);blocks=np.array([1,1,1,2,2,2])
        shuffled=s.permute_actor_categories(cats,blocks,np.random.default_rng(3))
        for fold in [1,2]:
            self.assertEqual(sorted(cats[blocks==fold]),sorted(shuffled[blocks==fold]))
        # One output per actor, rather than independent outputs for its videos.
        actor_rows=np.array([0,0,0,1,2,2,3,4,5,5])
        video_categories=shuffled[actor_rows]
        self.assertEqual(len(set(video_categories[:3])),1)

    def test_unknown_can_be_held_fixed_in_sensitivity(self):
        cats=np.array(['A','B','Unknown','A','C','C']);blocks=np.array([1,1,1,2,2,2])
        shuffled=s.permute_actor_categories(cats,blocks,np.random.default_rng(3),fixed_categories=('Unknown',))
        self.assertEqual(shuffled[2],'Unknown')
        self.assertEqual(sorted(shuffled[:2]),['A','B'])

    def test_holm_family_adjustment(self):
        np.testing.assert_allclose(s.holm_adjust([.01,.04,.03,.2]),[.04,.09,.09,.2])

    def test_historical_and_actor_aware_eligibility_are_separate_estimands(self):
        frame=pd.DataFrame({'model':['EmoT']*3,'attribute':['Race']*3,'group':['A','B','Unknown'],
            'n_videos':[100,60,82],'n_actors':[3,2,1],'accuracy':[.7,.65,.4],'macro_f1':[.69,.64,.39]})
        result=s.gap_eligibility_sensitivity(frame).set_index('eligibility_policy')
        self.assertAlmostEqual(result.loc['historical_videos_ge30','accuracy_gap'],.3)
        self.assertAlmostEqual(result.loc['known_groups_actors_ge2_videos_ge30','accuracy_gap'],.05)
        self.assertIn('Unknown',result.loc['historical_videos_ge30','groups'])
        self.assertNotIn('Unknown',result.loc['known_groups_actors_ge2_videos_ge30','groups'])

    def small_valid_inputs(self):
        m=pd.DataFrame({'ActorID':[10,10,20],'y':[0,1,2],'file_stem':['a','b','c'],'Race':['A','A','B'],'Sex':['F','F','M'],'Ethnicity':['X']*3,'age_bin':['20','20','30']})
        a=m.groupby('ActorID').first().reset_index()[['ActorID',*s.ATTRIBUTES]];a['outer_test_fold']=[1,2]
        rows=[]
        for model in ['EmoT','TemporalCNN1D']:
            for i in range(3):
                r={'sample_idx':i,'fold':1 if i<2 else 2,'model':model,'y_true':i,'y_pred':i}
                r.update({name:float(k==i) for k,name in enumerate(s.PROB_COLUMNS)});rows.append(r)
        return pd.DataFrame(rows),m,a

    def test_true_label_alignment_rejected(self):
        p,m,a=self.small_valid_inputs();p.loc[0,'y_true']=4
        with self.assertRaisesRegex(ValueError,'True labels disagree'):
            s.validate_inputs(p,m,a)

    def test_actor_fold_alignment_rejected(self):
        p,m,a=self.small_valid_inputs();p.loc[0,'fold']=2
        with self.assertRaisesRegex(ValueError,'Fold assignments disagree'):
            s.validate_inputs(p,m,a)

    def test_missing_sample_is_rejected(self):
        p,m,a=self.small_valid_inputs();p=p.drop(index=0)
        with self.assertRaisesRegex(ValueError,'covered once'):
            s.validate_inputs(p,m,a)

    def test_invalid_probability_is_rejected(self):
        p,m,a=self.small_valid_inputs();p.loc[0,'prob_angry']=.5
        with self.assertRaisesRegex(ValueError,'Invalid probabilities'):
            s.validate_inputs(p,m,a)

    def test_valid_alignment(self):
        p,m,a=self.small_valid_inputs();models,actors=s.validate_inputs(p,m,a,3,2)
        self.assertEqual(set(models),{'EmoT','TemporalCNN1D'});self.assertEqual(len(actors),2)


if __name__=='__main__':unittest.main()
