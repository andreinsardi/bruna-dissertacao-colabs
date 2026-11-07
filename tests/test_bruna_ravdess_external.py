import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bruna_ravdess_external import (
    parse_ravdess, expected_stems, validate_manifest,
    validate_checkpoint_normalization, conditional_actor_bootstrap,
    validate_external_alignment,
)


class RavdessExternalContracts(unittest.TestCase):
    def test_common_classes_and_modality(self):
        expected = {5: 0, 7: 1, 6: 2, 3: 3, 1: 4, 4: 5}
        for emotion, label in expected.items():
            row = parse_ravdess(f'02-01-{emotion:02d}-01-01-01-24.mp4')
            self.assertEqual(row['y'], label)
            self.assertEqual(row['ActorID'], 24)
        for name in ('02-01-02-01-01-01-01.mp4',
                     '02-01-08-01-01-01-01.mp4',
                     '01-01-05-01-01-01-01.mp4',
                     '02-02-05-01-01-01-01.mp4'):
            self.assertIsNone(parse_ravdess(name))

    def test_invalid_metadata_fails(self):
        for name in ('02-01-05-01-01-01-25.mp4',
                     '02-01-01-02-01-01-01.mp4',
                     '02-01-05-03-01-01-01.mp4',
                     '02-01-05-01-03-01-01.mp4'):
            with self.assertRaises(ValueError):
                parse_ravdess(name)

    def test_full_speech_recorte(self):
        stems = expected_stems()
        self.assertEqual(len(stems), 1056)
        manifest = pd.DataFrame([parse_ravdess(s+'.mp4') for s in sorted(stems)])
        validate_manifest(manifest)
        self.assertEqual(manifest.groupby('ActorID').size().unique().tolist(), [44])
        self.assertEqual(manifest.groupby('y').size().to_dict(),
                         {0: 192, 1: 192, 2: 192, 3: 192, 4: 96, 5: 192})

    def test_missing_and_duplicate_clips_rejected(self):
        frame = pd.DataFrame([parse_ravdess(s+'.mp4') for s in sorted(expected_stems())])
        with self.assertRaises(ValueError):
            validate_manifest(frame.iloc[:-1])
        with self.assertRaises(ValueError):
            validate_manifest(pd.concat([frame.iloc[:-1], frame.iloc[:1]]))

    def test_checkpoint_normalization_is_frozen_and_valid(self):
        payload = {'class_names': ['angry','disgust','fear','happy','neutral','sad'],
                   'mu': np.zeros((1,1,1240)), 'sigma': np.ones((1,1,1240))}
        mu, sigma = validate_checkpoint_normalization(payload)
        self.assertEqual(mu.shape, (1240,))
        self.assertTrue(np.all(sigma == 1))
        for field, value in [('sigma', np.zeros(1240)), ('mu', np.full(1240,np.nan)),
                             ('class_names', list(reversed(payload['class_names'])))]:
            invalid = dict(payload, **{field:value})
            with self.assertRaises(ValueError):
                validate_checkpoint_normalization(invalid)

    def test_bootstrap_paired_by_identity_and_deterministic(self):
        y = np.tile(np.arange(6), 3)
        actors = np.repeat(np.arange(3), 6)
        predictions = {'good': y, 'bad': (y+1)%6}
        result = conditional_actor_bootstrap(y, predictions, actors, n_boot=30, seed=9)
        repeat = conditional_actor_bootstrap(y, predictions, actors, n_boot=30, seed=9)
        pd.testing.assert_frame_equal(result, repeat)
        contrast = result.loc[result['contrast'].eq('good minus bad')]
        self.assertEqual(len(contrast), 2)
        self.assertTrue(np.allclose(contrast['estimate'], 1))
        self.assertTrue(np.allclose(contrast['ci_low'], 1))
        self.assertTrue(np.allclose(contrast['ci_high'], 1))
        self.assertTrue((result.n_actors == 3).all())

    def test_external_cache_alignment_rejects_reordered_manifest(self):
        stems=sorted(expected_stems())[:3]
        frame=pd.DataFrame([parse_ravdess(s+'.mp4') for s in stems])
        x=np.zeros((3,100,310),np.float32)
        cache={'X_base':x.copy(),'y':frame.y.to_numpy(), 'file_stems':np.asarray(stems),
               'selected_landmark_ids':np.arange(155)}
        validate_external_alignment(frame,x,cache,np.arange(155))
        for invalid in (frame.iloc[::-1], frame.assign(ActorID=24)):
            with self.assertRaises(ValueError):
                validate_external_alignment(invalid,x,cache,np.arange(155))
        changed=x.copy();changed[1,0,0]=1
        with self.assertRaises(ValueError):
            validate_external_alignment(frame,changed,cache,np.arange(155))

    def test_bootstrap_preserves_pairing_with_heterogeneous_actor_performance(self):
        y=np.tile(np.arange(6),4)
        actors=np.repeat(np.arange(4),6)
        predicted=y.copy()
        predicted[actors==0]=(y[actors==0]+1)%6
        predicted[actors==1]=0
        result=conditional_actor_bootstrap(y,{'a':predicted,'b':predicted.copy()},actors,n_boot=100,seed=17)
        contrast=result[result.contrast.eq('a minus b')]
        self.assertTrue(np.allclose(contrast[['estimate','ci_low','ci_high']],0))

    def test_extractor_functions_match_canonical_ast(self):
        import ast
        from build_bruna_ravdess_notebook import canonical_sources,CANONICAL_FUNCTIONS
        functions,_=canonical_sources()
        path=Path(__file__).resolve().parents[1]/'scripts/bruna_ravdess_canonical_extractor.py'
        generated={n.name:n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
        for name in CANONICAL_FUNCTIONS:
            self.assertEqual(ast.dump(generated[name],include_attributes=False),
                             ast.dump(ast.parse(functions[name]).body[0],include_attributes=False))

    def test_archive_download_checks_official_checksum_and_never_accepts_corrupt_bytes(self):
        import hashlib
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from bruna_ravdess_external import _download_archive
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):return None
            def raise_for_status(self):pass
            def iter_content(self,*args):yield b'verified-archive-bytes'
        content=b'verified-archive-bytes'
        record={'url':'https://zenodo.org/api/records/1188976/files/test.zip/content',
                'size':len(content),'checksum':'md5:'+hashlib.md5(content).hexdigest()}
        fake=SimpleNamespace(get=lambda *args,**kwargs:Response())
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,{'requests':fake}):
            target=Path(folder)/'actor.zip'
            _download_archive(record,target)
            self.assertEqual(target.read_bytes(),content)
            target.unlink()
            with self.assertRaises(RuntimeError):
                _download_archive(dict(record,checksum='md5:'+'0'*32),target)
            self.assertFalse(target.exists())

    def test_archive_download_rejects_nonofficial_host(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from bruna_ravdess_external import _download_archive
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,{'requests':SimpleNamespace()}):
            with self.assertRaises(ValueError):
                _download_archive({'url':'https://example.com/archive.zip'},Path(folder)/'actor.zip')


if __name__ == '__main__':
    unittest.main()
