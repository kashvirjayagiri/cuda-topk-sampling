"""CPU tests for the shared runner; optional actual PyTorch/CUDA tests on CARC."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))
from shared_comparison_utils import (
    assert_unchanged, finite_logits, load_dataset, preflight_cases, reference_table,
    shared_randoms, sliced_reference, timing_stats, validate_output, window_starts)
from shared_pipeline import METHODS, ComparisonLibrary
from summarize_shared_comparison import aggregate_rows, paired_rows, summarize_run

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ReferenceTests(unittest.TestCase):
    def test_stable_ties_and_original_ids(self):
        raw = np.array([[3,5,5,5,-1],[-2,-1,-1,-1,-3]], np.float32)
        refs = reference_table(raw, [1,3], [.5,1.,2.])
        np.testing.assert_array_equal(refs[3,1.]["ids"], [[1,2,3],[1,2,3]])
        np.testing.assert_array_equal(refs[1,1.]["ids"], [[1],[1]])
        np.testing.assert_allclose(refs[3,1.]["probabilities"], np.full((2,3),1/3))

    def test_hand_computed_softmax_and_temperature(self):
        raw = np.array([[0., np.log(3.)]], np.float32)
        ref = reference_table(raw,[2],[1.])[2,1.]
        np.testing.assert_allclose(ref["probabilities"], [[.75,.25]], atol=1e-7)
        np.testing.assert_array_equal(ref["ids"], [[1,0]])
        colder = reference_table(raw,[2],[.5])[2,.5]
        self.assertGreater(colder["probabilities"][0,0], ref["probabilities"][0,0])

    def test_extreme_fp32_stays_finite(self):
        raw=np.array([[np.finfo(np.float32).max,-np.finfo(np.float32).max,0.]],np.float32)
        ref=reference_table(raw,[3],[float(np.nextafter(np.float32(0),np.float32(1)))])
        for value in ref.values():
            self.assertTrue(np.isfinite(value["probabilities"]).all())
            self.assertEqual(value["probabilities"][0,0],1.)

    def test_exact_cdf_boundary_and_zero_probability(self):
        raw=np.zeros((2,4),np.float32)
        ref=reference_table(raw,[4],[1.])[4,1.]
        u=np.array([.5,.75],np.float32)
        expected=sliced_reference(ref,0,2,u)
        np.testing.assert_array_equal(expected["sampled_ids"],[2,3])
        self.assertEqual(validate_output(expected,deepcopy(expected),u)["max_cdf_error"],0.)

    def test_random_values_reproducible_and_away_from_union(self):
        raw=np.random.default_rng(5).normal(size=(32,101)).astype(np.float32)
        refs=reference_table(raw,[1,10,50,100],[.5,1.,2.])
        a=shared_randoms(refs,32,451,"a"*64)
        b=shared_randoms(refs,32,451,"a"*64)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.dtype,np.float32)
        self.assertTrue(np.all((a>=0)&(a<1)))
        for ref in refs.values(): self.assertGreater(np.min(np.abs(ref["cdf"]-a[:,None])),5e-5)

    def test_uniform_policy_does_not_change_draws(self):
        refs=reference_table(np.zeros((4,4),np.float32),[4],[1.])
        actual=shared_randoms(refs,4,451,"b"*64,0)
        expected=np.random.default_rng(np.random.SeedSequence([451,int("b"*8,16)])).random(4,dtype=np.float32)
        np.testing.assert_array_equal(actual,expected)

    def test_reject_bad_inputs(self):
        for raw in ([1,2], [[np.nan]], [[np.inf]], [[1+1j]], [["bad"]]):
            with self.subTest(raw=raw), self.assertRaises(ValueError): finite_logits(raw)
        with self.assertRaises(ValueError): reference_table(np.zeros((2,2)),[3],[1.])

    def test_saved_numpy_load(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"scores.npy"
            np.save(path,np.ones((3,19),np.float32))
            raw,info=load_dataset(path)
            self.assertEqual(raw.shape,(3,19))
            self.assertEqual(len(info["source_sha256"]),64)
            self.assertEqual(info["kind"],"saved_logits")

    def test_windows_cover_all_rows(self):
        for rows,batch in ((32,1),(32,8),(32,32),(33,8)):
            starts=window_starts(rows,batch)
            covered={i for start in starts for i in range(start,start+batch)}
            self.assertEqual(covered,set(range(rows)))
        self.assertEqual([len(window_starts(32,b)) for b in (1,8,32)],[32,4,1])
        self.assertEqual(window_starts(32,8,"2"),[0,8])
        with self.assertRaises(ValueError): window_starts(4,8)

    def test_mutation_detects_signed_zero_change(self):
        raw=np.array([[0.,-0.]],np.float32)
        assert_unchanged(raw.copy(),raw,"input")
        with self.assertRaises(ValueError): assert_unchanged(np.zeros_like(raw),raw,"input")

    def test_wrong_ids_and_samples_rejected(self):
        ref=reference_table(np.zeros((1,4),np.float32),[4],[1.])[4,1.]
        u=np.array([.5],np.float32)
        expected=sliced_reference(ref,0,1,u)
        for field in ("ids","sampled_ids","probabilities","cdf"):
            actual=deepcopy(expected)
            actual[field].flat[0]=99
            with self.subTest(field=field),self.assertRaises(ValueError): validate_output(expected,actual,u)

    def test_timing_statistics_and_bad_samples(self):
        self.assertEqual(timing_stats([3,1,2])["median_us"],2.)
        for values in ([],[0],[-1],[np.nan],[np.inf]):
            with self.assertRaises(ValueError): timing_stats(values)


def result_rows():
    rows=[]
    for start,fs,hier,py in ((0,100.,100.,120.),(1,1000.,200.,300.),(2,1000.,1000.,1100.)):
        for method,latency in zip(METHODS,(fs,hier,py)):
            rows.append(dict(source_sha256="c"*64,dataset="temporary_test_only",dataset_kind="smoke_synthetic",
                distribution="normal",B=1,V=19,k=10,temperature=1.,row_start=start,method=method,
                correctness="passed",timing_backend="graph",calls_per_sample=1,warmup=3,repetitions=5,
                workspace_bytes=128,selection_median_us=latency,pipeline_median_us=latency+1))
    return rows


class SummaryTests(unittest.TestCase):
    def test_pair_identical_cases_and_reject_missing_duplicate(self):
        rows=result_rows()
        paired=paired_rows(rows)
        self.assertEqual(len(paired),9)
        self.assertEqual(paired[0]["selection_speedup_vs_full_sort"],1.)
        with self.assertRaises(ValueError): paired_rows(rows[:-1])
        with self.assertRaises(ValueError): paired_rows(rows+[rows[0]])

    def test_aggregate_median_of_paired_speedups(self):
        result=aggregate_rows(paired_rows(result_rows()))
        hier=next(x for x in result if x["method"]=="hierarchical")
        self.assertEqual(hier["selection_median_paired_speedup_vs_full_sort"],1.)
        self.assertEqual(hier["window_count"],3)

    def test_summary_end_to_end_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            metadata=dict(status="complete",method_rows_written=9,timing_backend="graph",calls_per_sample=1,
                repetitions=5,warmup=3,gpu="TEMPORARY TEST FIXTURE",random_policy="test",random_margin=0)
            (root/"metadata.json").write_text(json.dumps(metadata))
            rows=result_rows()
            with (root/"summary.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
            paired,aggregates=summarize_run(root)
            self.assertEqual(len(paired),9)
            self.assertEqual(len(aggregates),3)
            self.assertTrue((root/"analysis_notes.md").is_file())
            metadata["status"]="incomplete"
            (root/"metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,"incomplete"): summarize_run(root)


@unittest.skipUnless(HAS_TORCH,"PyTorch is not installed")
class TorchTests(unittest.TestCase):
    def test_cpu_adapter_boundary_ties(self):
        import torch
        from torch_topk_adapter import CanonicalTopK
        rng=np.random.default_rng(18)
        with torch.no_grad():
            for k in (1,3,10,37):
                raw=rng.integers(-2,3,size=(3,37)).astype(np.float32)
                selector=CanonicalTopK(torch.from_numpy(raw.copy()),k)
                selector.select()
                expected=reference_table(raw,[k],[1.])[k,1.]
                np.testing.assert_array_equal(selector.ids.numpy(),expected["ids"])
                np.testing.assert_array_equal(selector.scores.numpy(),expected["scores"])

    def test_actual_gpu_integrations(self):
        import argparse
        import torch
        if not torch.cuda.is_available(): self.skipTest("No CUDA GPU")
        path=ROOT/"build/libshared_comparison_python.so"
        if not path.is_file(): self.skipTest("Build shared_comparison_python first")
        from run_shared_comparison import run_preflight
        torch.cuda.set_device(0)
        stream=torch.cuda.Stream(device=0)
        stream.wait_stream(torch.cuda.current_stream(0))
        args=argparse.Namespace(timing="graph",calls_per_sample=1)
        with torch.no_grad(),torch.cuda.stream(stream):
            checked=run_preflight(ComparisonLibrary(path),torch.device("cuda",0),stream,args)
        self.assertEqual(len(checked),48)


if __name__=="__main__": unittest.main()
