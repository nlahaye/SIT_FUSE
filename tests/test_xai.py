"""Lightweight XAI regressions; no model checkpoints, GPU, SHAP or TimeSHAP needed."""
import contextlib
import importlib.util
from pathlib import Path
import pickle
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


XAI_PATH = Path(__file__).resolve().parents[1] / "src/sit_fuse/xai/xai.py"


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def to(self, device):
        return self

    def float(self):
        return Tensor(self.values.astype(np.float32))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Model:
    num_classes = 2

    def __init__(self, torch_stub, temporal=False):
        self.torch = torch_stub
        self.temporal = temporal
        self.training = True
        self.batch_lengths = []
        self.clust_tree = {}
        self.pretrained_model = SimpleNamespace(to=lambda device: self.pretrained_model)

    def eval(self):
        self.training = False
        return self

    def to(self, device):
        self.device = device
        return self

    def forward(self, tensor, return_embed=False):
        assert not self.torch.grad_enabled
        assert not self.training
        self.batch_lengths.append(len(tensor.values))
        if self.temporal:
            score = tensor.values.mean(axis=(1, 2))
            return Tensor(np.stack([-score, score], axis=1))
        output = (tensor.values[:, :1] > 0).astype(np.float32)
        return None, Tensor(output), None, None

    __call__ = forward


def module(name, **attrs):
    mod = ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


class XAITest(unittest.TestCase):
    def setUp(self):
        self.torch = module(
            "torch", grad_enabled=True, nn=SimpleNamespace(Module=Model),
            from_numpy=Tensor, device=lambda name: name,
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(cudnn=SimpleNamespace()),
        )

        @contextlib.contextmanager
        def no_grad():
            previous = self.torch.grad_enabled
            self.torch.grad_enabled = False
            try:
                yield
            finally:
                self.torch.grad_enabled = previous

        def softmax(tensor, dim):
            values = tensor.values
            exp = np.exp(values - values.max(axis=dim, keepdims=True))
            return Tensor(exp / exp.sum(axis=dim, keepdims=True))

        self.torch.no_grad = no_grad
        self.torch.nn.functional = SimpleNamespace(softmax=softmax)
        self.shap = module("shap", KernelExplainer=Mock(), Explanation=SimpleNamespace)
        self.read_yaml = Mock()
        self.splits = Mock()
        self.load_temporal = Mock()
        self.dataset = Mock()
        self.get_model = Mock()
        self.calc = Mock()
        self.report = Mock()
        self.modules = {
            "torch": self.torch,
            "torch.optim": module("torch.optim"),
            "shap": self.shap,
            "timeshap": module("timeshap"),
            "timeshap.plot": module("timeshap.plot", plot_global_report=self.report),
            "timeshap.explainer": module("timeshap.explainer"),
            "timeshap.explainer.global_methods": module(
                "timeshap.explainer.global_methods", calc_global_explanations=self.calc),
            "sit_fuse.datasets.rtdbn_data_utils": module(
                "sit_fuse.datasets.rtdbn_data_utils", build_rtdbn_splits=self.splits),
            "sit_fuse.train.pretrain_rtdbn_dc": module(
                "sit_fuse.train.pretrain_rtdbn_dc", load_trained_rtdbn_dc=self.load_temporal),
            "sit_fuse.utils": module("sit_fuse.utils", read_yaml=self.read_yaml),
            "sit_fuse.inference.generate_output": module(
                "sit_fuse.inference.generate_output", get_model=self.get_model),
            "sit_fuse.datasets.dataset_utils": module(
                "sit_fuse.datasets.dataset_utils", get_prediction_dataset=self.dataset),
            "joblib": module("joblib", load=Mock()),
            "matplotlib": module("matplotlib", use=Mock()),
            "matplotlib.pyplot": module("matplotlib.pyplot"),
        }
        for name, cls in (("dbn_dc", "DBN_DC"), ("ijepa_dc", "IJEPA_DC"), ("dc", "DeepCluster")):
            path = f"sit_fuse.models.deep_cluster.{name}"
            self.modules[path] = module(path, **{cls: Mock()})
        with patch.dict(sys.modules, self.modules):
            spec = importlib.util.spec_from_file_location("xai_under_test", XAI_PATH)
            self.xai = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.xai)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def conf(self, encoder="dbn", **options):
        return {
            "encoder_type": encoder, "output": {"out_dir": str(self.out)},
            "cluster": {"num_classes": 2}, "rtdbn": {"n_visible": 2},
            "data": {"files_test": ["synthetic"], "fill_value": -1,
                     "number_channels": 2, "chan_dim": 0, "pixel_padding": 0},
            "xai": options,
        }

    def kernel(self, values=None):
        explainer = Mock(expected_value=np.array([0.0]))
        explainer.shap_values.side_effect = (
            (lambda data, **kwargs: [data.copy()]) if values is None
            else (lambda data, **kwargs: values))
        self.shap.KernelExplainer.return_value = explainer
        return explainer

    def test_config_precedence_defaults_and_validation(self):
        self.assertEqual(self.xai.xai_options({})["max_samples_per_cluster"], 200)
        options = self.xai.xai_options(
            {"xai": {"batch_size": 3, "max_evals": 64}},
            batch_size=2, max_evals=None)
        self.assertEqual(options["batch_size"], 2)
        self.assertEqual(options["max_evals"], 64)
        for key, value in (("batch_size", 0), ("max_evals", -1),
                           ("max_samples_per_cluster", -1),
                           ("max_windows_per_cluster", -1), ("seed", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.xai.xai_options({"xai": {key: value}})

    def test_sampling_retains_rare_clusters_and_is_reproducible(self):
        labels = np.array([0] * 20 + [1] * 2 + [2])
        selected = self.xai.sample_cluster_indices(labels, 3, seed=9)
        np.testing.assert_array_equal(
            selected, self.xai.sample_cluster_indices(labels, 3, seed=9))
        np.testing.assert_array_equal(np.bincount(labels[selected]), [3, 2, 1])
        np.testing.assert_array_equal(
            self.xai.sample_cluster_indices(labels, 0), np.arange(len(labels)))
        self.assertEqual(len(self.xai.sample_cluster_indices([], 3)), 0)

    def test_spatial_wrapper_batches_preserves_order_dtype_and_no_grad(self):
        model = Model(self.torch)
        head = Model(self.torch)
        model.clust_tree = {"1": {"head": head, "missing": None}}
        data = np.arange(-10, 10, dtype=np.float64).reshape(10, 2)[:, ::-1]
        predict = self.xai.make_batched_output_fn(model, "cpu", batch_size=3)
        actual = predict(data)
        np.testing.assert_array_equal(actual, (data[:, :1] > 0).astype(np.float32))
        self.assertEqual(actual.shape, (10, 1))
        self.assertEqual(model.batch_lengths, [3, 3, 3, 1])
        self.assertFalse(head.training)
        self.assertEqual(head.device, "cpu")
        self.assertTrue(self.torch.grad_enabled)
        self.assertEqual(predict(data[:0]).shape, (0, 1))

    def test_temporal_wrapper_matches_unbatched_probabilities(self):
        data = np.arange(-12, 12).reshape(6, 2, 2)
        model = Model(self.torch, temporal=True)
        score = self.xai.make_rtdbn_cluster_score_fn(model, 1, "cpu", batch_size=2)
        actual = score(data)
        full = self.xai.make_rtdbn_cluster_score_fn(model, 1, "cpu", batch_size=100)(data)
        np.testing.assert_allclose(actual, full)
        self.assertEqual(actual.shape, (6, 1))
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual(model.batch_lengths[:3], [2, 2, 2])
        self.assertEqual(score(data[:0]).shape, (0, 1))

    def test_explain_forwards_budget_and_saves_single_output_explanation(self):
        data = np.arange(6).reshape(3, 2)
        kernel = self.kernel()
        state = np.random.get_state()
        explanation = self.xai.explain(
            lambda x: x[:, :1], data, np.zeros((1, 2)), "identity",
            ["cluster"], str(self.out), nsamples=17, rs=4)
        kernel.shap_values.assert_called_once_with(data, nsamples=17)
        self.assertEqual(explanation.values.shape, (3, 2))
        self.assertEqual(explanation.base_values.shape, (3,))
        with (self.out / "explanation.pkl").open("rb") as f:
            np.testing.assert_array_equal(pickle.load(f).data, data)
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])

    def test_explain_normalizes_multiple_outputs(self):
        data = np.ones((3, 2))
        kernel = self.kernel([data, -data])
        kernel.expected_value = np.array([0.0, 1.0])
        explanation = self.xai.explain(
            lambda x: x, data, data[:1], "identity", ["a", "b"], str(self.out))
        self.assertEqual(explanation.values.shape, (3, 2, 2))
        self.assertEqual(explanation.base_values.shape, (3, 2))

    def test_spatial_entrypoint_sampling_artifacts_and_cache_invalidation(self):
        self.read_yaml.return_value = self.conf(max_samples_per_cluster=2, batch_size=3)
        data = np.arange(-12, 12, dtype=np.float32).reshape(12, 2)
        self.dataset.return_value = SimpleNamespace(data_full=data), None
        self.get_model.return_value = Model(self.torch)
        kernel = self.kernel()
        seen = []

        def plot(out_dir, values, samples, label, names):
            np.testing.assert_array_equal(values, samples)
            self.assertEqual(len(samples), 2)
            seen.append(label)
            (Path(out_dir) / f"shap_summary_plot_{label}.png").write_bytes(b"plot stub")

        with patch.object(self.xai, "save_summary_plot", side_effect=plot):
            self.xai.main(yaml="config", max_evals=19)
            self.xai.main(yaml="config", max_evals=19)
            self.assertEqual(kernel.shap_values.call_count, 1)
            self.xai.main(yaml="config", max_evals=23)
            self.assertEqual(kernel.shap_values.call_count, 2)
        selected = np.load(self.out / "explanation_sample_indices.npy")
        self.assertEqual(len(selected), 4)
        with (self.out / "explanation_kmeans_background.pkl").open("rb") as f:
            np.testing.assert_array_equal(pickle.load(f).data, data[selected])
        for name in ("shap_values_kmeans_background.npz", "explanation_settings.json",
                     "shap_summary_plot_0.png", "shap_summary_plot_1.png", "shap plots"):
            self.assertTrue((self.out / name).exists(), name)
        self.assertEqual(set(seen), {0, 1})
        self.load_temporal.assert_not_called()

    def test_spatial_default_sampling_and_tile_plot_alignment(self):
        self.read_yaml.return_value = self.conf()
        self.read_yaml.return_value["data"]["pixel_padding"] = 1
        data = np.ones((205, 18), dtype=np.float32)
        self.dataset.return_value = SimpleNamespace(data_full=data), None
        self.get_model.return_value = Model(self.torch)
        self.kernel()
        with patch.object(self.xai, "save_summary_plot") as plot:
            self.xai.main(yaml="config")
        self.assertEqual(len(np.load(self.out / "explanation_sample_indices.npy")), 200)
        values, samples = plot.call_args.args[1:3]
        self.assertEqual(values.shape, (1800, 2))
        self.assertEqual(samples.shape, values.shape)

    def setup_temporal(self):
        data = np.arange(-16, 16, dtype=np.float32).reshape(8, 2, 2)
        self.splits.return_value = None, None, SimpleNamespace(data_full=data)
        self.load_temporal.return_value = Model(self.torch, temporal=True)
        csv = SimpleNamespace(to_csv=lambda path, index: Path(path).write_text("shap\n0\n"))
        self.calc.return_value = None, csv, csv
        self.report.return_value = None, SimpleNamespace(
            save=lambda path: Path(path).write_text("<html>stub report</html>"))
        return data

    def test_rtdbn_entrypoint_sampling_budget_and_artifacts(self):
        data = self.setup_temporal()
        self.read_yaml.return_value = self.conf(
            "rtdbn", max_windows_per_cluster=2, max_evals=31, batch_size=3, seed=7)
        result = Path(self.xai.main(yaml="config"))
        self.assertEqual(result, self.out / "rtdbn_timeshap")
        self.assertEqual(self.calc.call_count, 2)
        for call in self.calc.call_args_list:
            score, long_data, _, event, feature = call.args
            self.assertEqual(long_data.shape, (4, 4))
            self.assertEqual(event, {"rs": 7, "nsamples": 31})
            self.assertEqual(feature, event)
            self.assertEqual(score(data[:3]).shape, (3, 1))
        for cluster in range(2):
            for suffix in ("event_shap.csv", "feature_shap.csv", "timeshap_report.html"):
                self.assertTrue((result / f"cluster_{cluster}_{suffix}").exists())
        self.assertEqual(len(np.load(result / "sample_indices.npy")), 4)
        self.assertLessEqual(max(self.load_temporal.return_value.batch_lengths), 3)
        self.get_model.assert_not_called()

    def test_cli_overrides_temporal_config_and_can_disable_sampling(self):
        self.setup_temporal()
        self.read_yaml.return_value = self.conf(
            "rtdbn", max_windows_per_cluster=1, max_evals=9, batch_size=100)
        argv = [str(XAI_PATH), "-y", "config", "--max-windows-per-cluster", "0",
                "--max-evals", "13", "--batch-size", "2", "--seed", "5"]
        with patch.dict(sys.modules, self.modules), patch.object(sys, "argv", argv):
            runpy.run_path(str(XAI_PATH), run_name="__main__")
        self.assertEqual(len(np.load(self.out / "rtdbn_timeshap/sample_indices.npy")), 8)
        for call in self.calc.call_args_list:
            self.assertEqual(call.args[3], {"rs": 5, "nsamples": 13})
        self.assertLessEqual(max(self.load_temporal.return_value.batch_lengths), 2)


if __name__ == "__main__":
    unittest.main()
