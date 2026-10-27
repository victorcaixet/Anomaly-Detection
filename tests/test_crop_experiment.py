import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from insulator_anomaly.crop_experiment import (
    CropSettings, segment_boxes, padded_box, prepare_crop_experiment,
    aggregate_sources, compare_source_scores,
)


class CropExperimentTests(unittest.TestCase):
    def test_windows_cover_every_pixel_horizontal_vertical_square(self):
        for width,height in [(1000,80),(80,1000),(80,80),(101,37)]:
            covered = np.zeros((height,width),dtype=bool)
            for a,b,c,d in segment_boxes(width,height):
                covered[b:d,a:c] = True
            self.assertTrue(covered.all())
        self.assertEqual(padded_box((10,20,30,40),100,100,.5),(0,10,40,50))

    def test_preparation_pairs_sources_preserves_pixels_and_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rng=np.random.default_rng(2)
            for folder,count in [('train/good',5),('test/good',2),('test/anomaly',2)]:
                target=root/'source'/folder
                target.mkdir(parents=True)
                for i in range(count):
                    cv2.imwrite(str(target/f'{i}.png'),rng.integers(0,256,(32,128,3),dtype=np.uint8))
            manifest=prepare_crop_experiment(root/'source',root/'run',CropSettings(output_size=128))
            whole=manifest[manifest['mode']=='whole']
            segments=manifest[manifest['mode']=='segments']
            self.assertEqual(set(whole.source_id),set(segments.source_id))
            self.assertEqual(set(whole[whole.split=='train'].source_id) & set(whole[whole.split=='val'].source_id),set())
            self.assertFalse(manifest[manifest.split.isin(['train','val'])].y_true.any())
            sample=whole.iloc[0]
            original=cv2.imread(str(root/'source'/sample.source_id))
            output=cv2.imread(sample.image_path)
            np.testing.assert_array_equal(output[48:80],original)
            with self.assertRaises(FileExistsError):
                prepare_crop_experiment(root/'source',root/'run')

    def test_duplicates_are_grouped_before_holdout_and_test_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('train/good', 'test/good', 'test/anomaly'):
                (root/'source'/folder).mkdir(parents=True)
            for name, value in [('a', 1), ('b', 1), ('c', 2), ('d', 3)]:
                cv2.imwrite(str(root/'source/train/good'/f'{name}.png'),
                            np.full((16, 32, 3), value, dtype=np.uint8))
            for folder, value in [('good', 4), ('anomaly', 5)]:
                cv2.imwrite(str(root/'source/test'/folder/'x.png'),
                            np.full((16, 32, 3), value, dtype=np.uint8))
            # Explicit groups linked by a duplicate must merge as a whole.
            groups = {'train/good/b.png': 'sequence', 'train/good/c.png': 'sequence'}
            manifest = prepare_crop_experiment(root/'source', root/'run', groups=groups)
            normal = manifest[manifest.source_id.isin(
                ['train/good/a.png', 'train/good/b.png', 'train/good/c.png'])]
            self.assertEqual(normal.split.nunique(), 1)
            self.assertEqual(normal.group_id.nunique(), 1)
            self.assertTrue((manifest.groupby('source_sha256').split.nunique() == 1).all())
            (root/'source/test/good/x.png').write_bytes(
                (root/'source/train/good/a.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'train/good/a.png.*test/good/x.png'):
                prepare_crop_experiment(root/'source', root/'blocked')
            self.assertFalse((root/'blocked').exists())

            # Removing a complete connected group must still leave enough data.
            with self.assertRaisesRegex(ValueError, 'dois grupos'):
                prepare_crop_experiment(root/'source', root/'too_small', groups=groups,
                                        duplicate_policy='exclude_train')
            self.assertFalse((root/'too_small').exists())
            cv2.imwrite(str(root/'source/train/good/e.png'),
                        np.full((16, 32, 3), 6, dtype=np.uint8))
            cleaned = prepare_crop_experiment(root/'source', root/'clean', groups=groups,
                                              duplicate_policy='exclude_train')
            self.assertTrue((cleaned.groupby('source_sha256').split.nunique() == 1).all())
            self.assertEqual(set(cleaned[cleaned.split == 'test'].source_id),
                             {'test/good/x.png', 'test/anomaly/x.png'})
            excluded = pd.read_csv(root/'clean/excluded_sources.csv')
            self.assertEqual(set(excluded.source_id),
                             {'train/good/a.png', 'train/good/b.png', 'train/good/c.png'})
            for name in ('a', 'b', 'c'):
                self.assertTrue((root/'source/train/good'/f'{name}.png').exists())
            # Label conflicts cannot be resolved by excluding training sources.
            (root/'source/test/anomaly/x.png').write_bytes(
                (root/'source/test/good/x.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'rótulos'):
                prepare_crop_experiment(root/'source', root/'conflict', groups=groups,
                                        duplicate_policy='exclude_train')
            self.assertFalse((root/'conflict').exists())

    def test_aggregation_and_normal_only_threshold(self):
        manifest=pd.DataFrame([
            dict(mode='segments',split='test',source_id='bad',group_id='bad',y_true=1,image_path='a'),
            dict(mode='segments',split='test',source_id='bad',group_id='bad',y_true=1,image_path='b')])
        scores=aggregate_sources(manifest,pd.DataFrame({'image_path':['a','b'],'pred_score':[.1,.8]}))
        self.assertEqual(scores.iloc[0].pred_score,.8)
        self.assertEqual(scores.iloc[0].n_crops,2)
        rows=[]
        for mode in ['whole','segments']:
            for split,source,label,score in [('val','v1',0,.1),('val','v2',0,.2),('test','good',0,.15),('test','bad',1,.8)]:
                rows.append(dict(mode=mode,split=split,source_id=source,group_id=source,y_true=label,pred_score=score))
        comparison,_=compare_source_scores(pd.DataFrame(rows))
        self.assertTrue((comparison.threshold==.2).all())
        self.assertTrue((comparison.TP==1).all())
        self.assertTrue((comparison.FP==0).all())
        changed=pd.DataFrame(rows)
        changed.loc[(changed.split=='test') & (changed.y_true==1),'pred_score']=100
        other,_=compare_source_scores(changed)
        np.testing.assert_array_equal(comparison.threshold,other.threshold)


if __name__=='__main__':
    unittest.main()
