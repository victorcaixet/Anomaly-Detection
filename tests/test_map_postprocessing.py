import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from insulator_anomaly.map_postprocessing import (
    MapSettings,mebin_mask,local_contrast,conformal_pvalues,evaluate_map_methods,
)


class MapPostprocessingTests(unittest.TestCase):
    def test_constant_and_low_absolute_peak(self):
        empty,threshold=mebin_mask(np.full((32,32),.2))
        self.assertFalse(empty.any())
        array=np.full((32,32),.1)
        array[12:16,12:16]=.41
        mask,threshold=mebin_mask(array)
        self.assertTrue(mask[13,13])
        self.assertFalse(mask[0,0])
        self.assertLess(threshold,.51225)
        contrast=local_contrast(array,windows=(9,),scale_floor=.01)
        self.assertGreater(contrast[13,13],3)
        self.assertEqual(contrast[0,0],0)

    def test_conformal_ties_and_finite_sample_resolution(self):
        np.testing.assert_allclose(conformal_pvalues([1,2,2],[0,2,3]),[1,.75,.25])
        self.assertGreater(conformal_pvalues(np.zeros(10),[100])[0],.05)

    def _fixture(self,root):
        rows=[]
        rng=np.random.default_rng(3)
        for split,label,count in [('reference',0,3),('calibration',0,5),('test',0,2),('test',1,2)]:
            for i in range(count):
                key=f'{split}_{label}_{i}'
                array=rng.uniform(.09,.11,(24,24))
                if label:
                    array[8:12,8:12]=.4
                path=root/f'{key}.npy'
                np.save(path,array)
                cv2.imwrite(str(root/f'{key}.png'),np.full((24,24,3),128,dtype=np.uint8))
                rows.append(dict(mode='gray',split=split,source_id=key,group_id=key,
                                 y_true=label,image_path=str(root/f'{key}.png'),map_path=str(path)))
        return pd.DataFrame(rows)

    def test_full_export_and_no_test_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frame=self._fixture(root)
            settings=MapSettings(windows=(9,),alpha=.2)
            comparison,metrics=evaluate_map_methods(frame,root/'first',settings,show_first=0)
            self.assertEqual(set(comparison.method),{'absolute','mebin','contrast'})
            self.assertTrue((root/'first'/'source_decisions.csv').exists())
            for path in frame.loc[frame.split=='test','map_path']:
                np.save(path,np.load(path)*20)
            evaluate_map_methods(frame,root/'second',settings,show_first=0)
            import json
            one=json.loads((root/'first'/'protocol.json').read_text())
            two=json.loads((root/'second'/'protocol.json').read_text())
            self.assertEqual(one['modes'],two['modes'])
            with self.assertRaises(FileExistsError):
                evaluate_map_methods(frame,root/'first',settings,show_first=0)
            frame.loc[frame.split=='reference','y_true']=1
            with self.assertRaises(ValueError):
                evaluate_map_methods(frame,root/'bad',settings,show_first=0)

    def test_group_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frame=self._fixture(root)
            frame.loc[0,'group_id']=frame.loc[frame.split=='test','group_id'].iloc[0]
            with self.assertRaises(ValueError):
                evaluate_map_methods(frame,root/'bad',show_first=0)

    def test_optional_annotations_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frame=self._fixture(root)
            frame['defect_path']=''
            for index,row in frame[frame.split=='test'].iterrows():
                defect=np.zeros((24,24),dtype=np.uint8)
                if row.y_true:
                    defect[8:12,8:12]=255
                path=root/f'defect_{index}.png'
                cv2.imwrite(str(path),defect)
                frame.loc[index,'defect_path']=str(path)
            _,table=evaluate_map_methods(frame,root/'out',MapSettings(windows=(9,),alpha=.2),show_first=1)
            self.assertTrue((root/'out'/'annotated_pixel_summary.csv').exists())
            self.assertEqual(len(list((root/'out'/'gray').glob('*comparison.png'))),1)
            annotated=table[(table.split=='test') & (table.y_true==1) & (table.method=='mebin')]
            self.assertTrue((annotated.iou==1).all())


if __name__=='__main__':
    unittest.main()
