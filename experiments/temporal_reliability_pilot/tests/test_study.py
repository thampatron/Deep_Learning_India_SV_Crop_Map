"""Regression tests: timestamps, split leakage, source separation, OOF and student."""
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from make_synthetic_sequence import generate
from study_utils import (load_inputs,spatial_partition,block_ids,month_distance,
                         temporal_features)
from run_study import (model_probe,reliability_probe,distillation_probe,stress_test)


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta,cls.sat,cls.obs,cls.ym = generate(n=700,seed=31)
        cls.g = cls.meta[[f"g_{i}" for i in range(6)]].to_numpy(float)

    def test_interval_month_offset_across_year(self):
        ym = np.array([[202312,202401,202402]])
        self.assertEqual(month_distance(ym,np.array([[202401]])).tolist(),
                         [[-1,0,1]])

    def test_spatial_groups_do_not_overlap(self):
        split=spatial_partition(self.meta,1.,0.,seed=31)
        groups=block_ids(self.meta,1.)
        gtrain=set(groups[split["train"]])
        gval=set(groups[split["val"]])
        gtest=set(groups[split["test"]])
        self.assertTrue(gtrain.isdisjoint(gval))
        self.assertTrue(gtrain.isdisjoint(gtest))
        self.assertTrue(gval.isdisjoint(gtest))
        self.assertTrue((self.meta.iloc[split["test"]].label_source=="expert").all())

    def test_temporal_alignment_uses_capture_year_and_month(self):
        small=self.meta.iloc[:3].copy()
        ground=self.g[:3]
        sat=np.nan_to_num(self.sat[:3],nan=0)
        observed=self.obs[:3]
        base=temporal_features(small,ground,sat,observed,self.ym[:3],"aligned")
        moved=small.copy()
        moved["capture_month"] = (moved.capture_month % 10) + 1
        changed=temporal_features(moved,ground,sat,observed,self.ym[:3],"aligned")
        self.assertFalse(np.allclose(base,changed))

    def test_corrupt_satellite_zero_observations_allowed(self):
        meta=self.meta.iloc[:5].copy()
        sat=np.nan_to_num(self.sat[:5],nan=0)
        observed=np.zeros_like(self.obs[:5])
        x=temporal_features(meta,self.g[:5],sat,observed,self.ym[:5],"aligned")
        self.assertTrue(np.isfinite(x).all())

    def test_csv_sequence_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            self.meta.to_csv(p/"metadata.csv",index=False)
            np.savez(p/"sequences.npz",satellite=self.sat,observed=self.obs,yearmonth=self.ym)
            meta,g,sat,obs,ym=load_inputs(p/"metadata.csv",p/"sequences.npz")
            self.assertEqual(len(meta),len(self.meta))
            self.assertEqual(sat.shape,(700,10,4))
            self.assertTrue(np.isfinite(sat).all())

    def test_full_scientific_pipeline_smoke(self):
        meta=self.meta
        sat=np.nan_to_num(self.sat,nan=0)
        split=spatial_partition(meta,1.,0.,seed=31)
        models,fitted,labels,gold=model_probe(meta,self.g,sat,self.obs,self.ym,split,seed=31)
        self.assertIn("aligned_shuffled_timestamp",models)
        self.assertIn("satellite_plus_date",models)
        self.assertTrue(0<=models["aligned"]["macro_f1"]<=1)
        reliability=reliability_probe(meta,self.g,sat,self.obs,self.ym,
                                      split,fitted,labels,gold)
        self.assertIn("oof_risk_weight",reliability)
        student=distillation_probe(meta,self.g,sat,self.obs,self.ym,split,labels,31)
        self.assertIn("oof_teacher_soft_student",student)
        stress=stress_test(meta,self.g,sat,self.obs,self.ym,split,fitted,labels,31)
        self.assertIn("wrong_field_0.3",stress["models"]["aligned"])
        self.assertIn("missing_sat_dates_0.6",stress["models"]["satellite"])


if __name__=="__main__":
    unittest.main()
