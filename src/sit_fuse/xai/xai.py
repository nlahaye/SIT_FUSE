"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

# General Imports
import argparse
import hashlib
import json
from timeshap.plot import plot_global_report
from timeshap.explainer.global_methods import calc_global_explanations
import shap
from sit_fuse.datasets.rtdbn_data_utils import build_rtdbn_splits
from sit_fuse.train.pretrain_rtdbn_dc import load_trained_rtdbn_dc
from sit_fuse.utils import read_yaml
from sit_fuse.inference.generate_output import get_model
from sit_fuse.datasets.dataset_utils import get_prediction_dataset
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dc import DeepCluster
import torch.optim as opt
import torch
from joblib import load
import pickle
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Serialization


# ML Imports


# from torch.nn.parallel import DistributedDataParallel as DDP

# Input Parsing

SEED = 42

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# TODO: Image plotting features
# TODO: Waterfall plot for average of observations over a label
# TODO: Parallel process shap values


def load_model(yml_conf, n_visible=None):

    num_loader_workers = int(yml_conf["data"]["num_loader_workers"])
    val_percent = int(yml_conf["data"]["val_percent"])
    batch_size = yml_conf["cluster"]["training"]["batch_size"]
    use_gpu = yml_conf["encoder"]["training"]["use_gpu"]
    out_dir = yml_conf["output"]["out_dir"]

    save_dir = yml_conf["output"]["out_dir"]
    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    if use_wandb_logger:
        save_dir = os.path.join(
            yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
    ckpt_path = os.path.join(os.path.join(
        save_dir, "full_model"), "deep_cluster.ckpt")
    ckpt_path_heir = os.path.join(os.path.join(
        save_dir, "full_model_heir"), "full_model_heir.ckpt")

    model = None

    if os.path.exists(ckpt_path_heir):
        model = Heir_DC.load_from_checkpoint(ckpt_path_heir)
    else:
        if "encoder_type" in yml_conf:
            if yml_conf["encoder_type"] == "dbn":
                model = DBN_DC.load_from_checkpoint(ckpt_path)
            elif yml_conf["encoder_type"] == "ijepa":
                model = IJEPA_DC.load_from_checkpoint(ckpt_path)
        else:
            model = DeepCluster.load_from_checkpoint(ckpt_path)

    return final_model


def wrap(x):
    """
    Reshapes from (channel, row, col) to (# pixels, channel) or (N_samples, N_features)
    """
    x = np.array(x)
    orig_shape = x.shape
    x = np.moveaxis(x, 0, 2)
    x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
    print(f"Wrapped {orig_shape} to {x.shape}.")
    return x


def unwrap(x, shape):
    """
    Reshapes from (N_samples, N_features) to (N_channels, N_rows, N_cols)
    Args:
        x: dataset (N_samples, N_features)
        shape: desired output shape (N_channels, N_rows, N_cols)
    Returns:
        Reshaped array.
    """
    # TODO: Support chan_dim != 0
    orig_shape = x.shape
    print(orig_shape)
    x = np.reshape(x, (shape[1], shape[2], shape[0]))
    x = np.moveaxis(x, 2, 0)
    print(f"Unwrapped {orig_shape} to {x.shape}.")
    return x


def filled_array(x, fill=0.):
    if isinstance(x, np.ndarray) or isinstance(x, list):
        return (np.zeros_like(x, dtype=x.data_full.dtype) + fill)
    elif isinstance(x, tuple):
        return (np.zeros(x, dtype=np.float32) + fill)
    else:
        return None


def forward_numpy(x, model):
    """Performs a forward pass over the data.

    Args:
        x: An input np.ndarray that will be converted to a tensor for computing the forward pass.

    Returns:
        (np.ndarray): An array containing the DBN's outputs.
    """
    numpy_to_torch_dtype_dict = {
        np.dtype(np.bool): torch.bool,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.complex64): torch.complex64,
        np.dtype(np.complex128): torch.complex128
    }

    dt = numpy_to_torch_dtype_dict[x.dtype]
    t = torch.from_numpy(x).cuda()
    y = model.forward(t)
    y = y.detach().cpu().numpy()

    return y


def most_frequent_labels(data):
    """
    Args: 
        data: (# samples, # labels)
    Returns: 
        Array of labels (# labels) sorted by frequency of occurence
    """

    if data.shape[1] > 1:
        labels = np.argmax(data, axis=1)
    else:
        labels = data.astype(np.int32)
    unique_labels, frequency = np.unique(labels, return_counts=True)
    sorted_indexes = np.argsort(frequency)[::-1]
    sorted_by_freq = unique_labels[sorted_indexes]

    return sorted_by_freq


def dims_from_indices(indices, n_channels, chan_dim=0):
    t = np.moveaxis(np.transpose(indices), chan_dim, 0)
    dims = (n_channels, np.max(t[1]) + 1, np.max(t[2]) + 1)
    np.moveaxis(dims, 0, chan_dim)
    return dims


def get_background(background_type, data: np.ndarray, n_samples=None):
    if background_type == "kmeans":
        background = shap.kmeans(data, n_samples)
    elif background_type == "sample":
        background = shap.sample(data, n_samples)
    elif background_type == "zero":
        background = np.zeros((1, data.shape[1]), dtype=data.dtype)
    else:
        background = data
    return background


def save_summary_plot(out_dir, shap_values, X, label, feature_names):
    filename = f"shap_summary_plot_{label}.png"
    print(f"Saving {os.path.join(out_dir, filename)}...")
    print(feature_names)
    p = shap.plots.violin(shap_values, features=X,
                          feature_names=feature_names, plot_type="layered_violin")
    # p = shap.summary_plot(shap_values, show=False, color_bar=True, feature_names=feature_names) #feature_names=feature_names, show=False, color_bar=True)
    plt.savefig(os.path.join(out_dir, filename),
                bbox_inches='tight', pad_inches=0.2, dpi=400)
    plt.clf()
    plt.cla()
    plt.close()


def save_shap(shap_values, filename):
    # Save all SHAP values
    with open(filename, 'wb') as f:
        pickle.dump(shap_values, f)


def explain(f, dataset: np.ndarray, background: np.ndarray, link, output_names, out_dir, explanation_fname="explanation.pkl", nsamples="auto", rs=SEED):

    print("Calculating shap values...")
    if link not in ['identity', 'logit']:
        link = 'identity'
    explainer = shap.KernelExplainer(
        f, background, link=link)
    print(dataset.shape, dataset.min(), dataset.max(), dataset.mean())
    # KernelExplainer's callable API does not forward the coalition budget.
    random_state = np.random.get_state()
    try:
        np.random.seed(rs)
        values = explainer.shap_values(dataset, nsamples=nsamples)
    finally:
        np.random.set_state(random_state)
    if isinstance(values, list):
        values = values[0] if len(values) == 1 else np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    base_shape = (len(dataset),) if values.ndim == 2 else (len(dataset), values.shape[-1])
    explanation = shap.Explanation(
        values=values,
        base_values=np.broadcast_to(np.asarray(explainer.expected_value).squeeze(), base_shape),
        data=dataset,
        output_names=list(output_names) if values.ndim == 3 else None,
    )
    save_shap(explanation, os.path.join(out_dir, explanation_fname))

    return explanation


def rtdbn_windows_to_timeshap_long(data3d: np.ndarray):
    """Reshapes (N, seq_len, n_visible) windows into TimeSHAP's long-format
    2D array (entity_col + time_col + features) -- avoids a crash in
    TimeSHAP's global explainer when entity_col=None."""
    n_windows, seq_len, n_features = data3d.shape
    entity_ids = np.repeat(np.arange(n_windows), seq_len)
    time_ids = np.tile(np.arange(seq_len), n_windows)
    flat_features = data3d.reshape(n_windows * seq_len, n_features)
    data2d = np.concatenate(
        [entity_ids[:, None], time_ids[:, None], flat_features], axis=1
    ).astype(np.float64)
    schema = ["entity", "time"] + [f"feat_{j}" for j in range(n_features)]
    return data2d, schema


def xai_options(yml_conf, **overrides):
    options = {
        "max_samples_per_cluster": 200,
        "max_windows_per_cluster": 200,
        "batch_size": 512,
        "max_evals": None,
        "seed": SEED,
    }
    options.update({key: value for key, value in yml_conf.get("xai", {}).items()
                    if key in options})
    options.update({key: value for key, value in overrides.items()
                    if key in options and value is not None})
    for key in options:
        if options[key] is not None:
            options[key] = int(options[key])
    for key in ("max_samples_per_cluster", "max_windows_per_cluster"):
        if options[key] < 0:
            raise ValueError(f"xai.{key} must be nonnegative (0 means all).")
    for key in ("batch_size", "max_evals"):
        if options[key] is not None and options[key] <= 0:
            raise ValueError(f"xai.{key} must be positive.")
    if options["seed"] < 0 or options["seed"] >= 2**32:
        raise ValueError("xai.seed must be in [0, 2**32).")
    return options


def sample_cluster_indices(labels, max_per_cluster, seed=SEED):
    """Keep rare clusters and deterministically sample larger clusters; 0 keeps all."""
    if max_per_cluster < 0:
        raise ValueError("max_per_cluster must be nonnegative.")
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        inds = np.flatnonzero(labels == label)
        if max_per_cluster and len(inds) > max_per_cluster:
            inds = rng.choice(inds, size=max_per_cluster, replace=False)
        selected.extend(inds)
    return np.sort(np.asarray(selected, dtype=np.int64))


def prepare_explanation_model(model, device):
    model.eval()
    # Hierarchical heads live in ordinary dictionaries, not registered modules.
    for tier in getattr(model, "clust_tree", {}).values():
        for head in tier.values():
            if isinstance(head, torch.nn.Module):
                head.to(device)
                head.eval()


def make_batched_output_fn(model, device, batch_size=512, temporal=False):
    """Bound device memory while retaining hierarchical labels or RTDBN probabilities."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    prepare_explanation_model(model, device)

    def f(x):
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                t = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(device)
                if temporal:
                    output = torch.nn.functional.softmax(model(t.float()), dim=1)
                else:
                    _, output, _, _ = model.forward(t, return_embed=False)
                outputs.append(output.detach().cpu().numpy())
        if not outputs:
            width = model.num_classes if temporal else 1
            return np.empty((0, width), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
    return f


def make_rtdbn_cluster_score_fn(model, cluster_id, device, batch_size=512):
    """Wraps RTDBN_DC's (N, num_classes) output into the (N, 1) per-cluster
    score function TimeSHAP requires. Double-softmaxes to match generate_output_rtdbn()."""
    predict = make_batched_output_fn(model, device, batch_size, temporal=True)

    def f(x):
        return predict(x)[:, cluster_id:cluster_id + 1].astype(np.float64)
    return f


def explain_rtdbn(yml_conf, max_windows_per_cluster=None, nsamples=None, rs=None, batch_size=None):
    """Mirrors main()'s per-cluster loop;
    writes CSVs and HTML reports to out_dir/rtdbn_timeshap/. Requires shap<=0.42.1
    (see pyproject.toml)."""
    options = xai_options(yml_conf, max_windows_per_cluster=max_windows_per_cluster,
                          max_evals=nsamples, seed=rs, batch_size=batch_size)
    max_windows_per_cluster = options["max_windows_per_cluster"]
    nsamples = options["max_evals"] or 500
    rs = options["seed"]
    batch_size = options["batch_size"]
    out_dir = yml_conf["output"]["out_dir"]
    encoder_dir = os.path.join(out_dir, "encoder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_rtdbn_dc(yml_conf)
    model = model.to(device)
    model.pretrained_model = model.pretrained_model.to(device)

    _, _, test_ds = build_rtdbn_splits(
        yml_conf, scaler_in_path=os.path.join(encoder_dir, "scaler.pkl")
    )
    if test_ds is None:
        raise ValueError(
            "No held-out test split configured (data.test_percent) -- TimeSHAP explanations need held-out windows.")

    num_classes = yml_conf["cluster"]["num_classes"]
    n_visible = int(yml_conf["rtdbn"]["n_visible"])

    all_windows = test_ds.data_full
    if len(all_windows) == 0:
        raise ValueError("No held-out windows available for TimeSHAP.")
    predict = make_batched_output_fn(model, device, batch_size, temporal=True)
    predicted_labels = np.argmax(predict(all_windows), axis=1)

    baseline = np.zeros((1, 1, n_visible))
    shap_out_dir = os.path.join(out_dir, "rtdbn_timeshap")
    os.makedirs(shap_out_dir, exist_ok=True)

    selected = sample_cluster_indices(predicted_labels, max_windows_per_cluster, rs)
    np.save(os.path.join(shap_out_dir, "sample_indices.npy"), selected)
    for cluster_id in range(num_classes):
        inds = selected[predicted_labels[selected] == cluster_id]
        if len(inds) == 0:
            print(f"No windows assigned to cluster {cluster_id}, skipping.")
            continue
        windows = all_windows[inds]
        data2d, schema = rtdbn_windows_to_timeshap_long(windows)
        model_features = schema[2:]

        f = make_rtdbn_cluster_score_fn(model, cluster_id, device, batch_size)

        event_dict = {"rs": rs, "nsamples": nsamples}
        feature_dict = {"rs": rs, "nsamples": nsamples}

        print(f"Explaining cluster {cluster_id} ({len(inds)} windows)...")
        # pruning=None: TimeSHAP's pruning has an edge-case bug (asserts
        # pruning_idx < seq_len); not needed for these short windows anyway.
        _, event_data, feat_data = calc_global_explanations(
            f, data2d, None, event_dict, feature_dict,
            baseline=baseline, model_features=model_features, schema=schema,
            entity_col="entity", time_col="time",
        )

        event_data.to_csv(os.path.join(
            shap_out_dir, f"cluster_{cluster_id}_event_shap.csv"), index=False)
        feat_data.to_csv(os.path.join(
            shap_out_dir, f"cluster_{cluster_id}_feature_shap.csv"), index=False)

        _, plot = plot_global_report(
            None, event_dict, feature_dict, None, event_data, feat_data)
        plot_path = os.path.join(
            shap_out_dir, f"cluster_{cluster_id}_timeshap_report.html")
        plot.save(plot_path)
        print(f"Saved cluster {cluster_id} TimeSHAP report to {plot_path}")

    print(f"RTDBN TimeSHAP explanations written to {shap_out_dir}")
    return shap_out_dir


def main(**kwargs):

    # TODO: Finish argparse params
    if 'yaml' in kwargs:
        yml_conf = read_yaml(kwargs['yaml'])
    else:
        raise Exception('No yaml path specified.')
    if 'masker' in kwargs:
        masker = kwargs['masker']
    else:
        masker = 'uniform_fill'
    options = xai_options(yml_conf, **kwargs)

    if yml_conf.get("encoder_type") == "rtdbn":
        # RTDBN needs TimeSHAP, not the spatial KernelExplainer path below.
        return explain_rtdbn(
            yml_conf, max_windows_per_cluster=options["max_windows_per_cluster"],
            nsamples=options["max_evals"], rs=options["seed"],
            batch_size=options["batch_size"],
        )

    out_dir = yml_conf['output']['out_dir']
    clusters = yml_conf['cluster']['num_classes']
    fill_value = yml_conf['data']['fill_value']
    n_channels = yml_conf['data']['number_channels']
    chan_dim = yml_conf['data']['chan_dim']

    # loop over multiple files + subsample
    # Load train and test data
    data, _ = get_prediction_dataset(
        yml_conf, yml_conf["data"]["files_test"][0])

    # Load model and scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(yml_conf, data.data_full.shape[1]).to(device)
    get_output = make_batched_output_fn(model, device, options["batch_size"])
    if len(data.data_full) == 0:
        raise ValueError("No samples available for SHAP.")
    output = get_output(data.data_full)
    predicted_labels = (np.argmax(output, axis=1) if output.shape[1] > 1
                        else output.reshape(-1))
    selected = sample_cluster_indices(
        predicted_labels, options["max_samples_per_cluster"], options["seed"])
    dat = data.data_full[selected]
    output = output[selected]
    os.makedirs(out_dir, exist_ok=True)

    # Subset? (yes: (0:200, 0:600))
    # row_min, row_max = 0, 200
    # col_min, col_max = 0, 600
    # new_dims = np.moveaxis(np.array([n_channels, row_max, col_max]), 0, chan_dim)

    # if row_min and col_min and row_max and col_max:
    #    data_train = data_train[:,row_min:row_max,col_min:col_max]
    #    data_test = data_test[:,row_min:row_max,col_min:col_max]

    # print("Data: ", data_test.shape)

    # Preprocess data
    # data_train = scaler.transform(wrap(data_train))
    # data_test = scaler.transform(wrap(data_test))
    # print("Model input: ", data_test.shape)

    # Set explain params
    overwrite = False  # True
    link = "identity"  # 'logit'
    background_type = 'zero'
    background = get_background(background_type, dat)

    # ============================== EXPLANATION ===============================

    labels = [x for x in range(clusters)]
    label_names = map(str, labels)

    explanation_path = os.path.join(out_dir, "explanation_kmeans_background.pkl")
    values_path = os.path.join(out_dir, "shap_values_kmeans_background.npz")
    indices_path = os.path.join(out_dir, "explanation_sample_indices.npy")
    settings_path = os.path.join(out_dir, "explanation_settings.json")
    settings = dict(options, input_shape=list(data.data_full.shape),
                    input_file=yml_conf["data"]["files_test"][0],
                    input_dtype=dat.dtype.str,
                    input_digest=hashlib.sha256(np.ascontiguousarray(dat).tobytes()).hexdigest(),
                    config_digest=hashlib.sha256(
                        json.dumps(yml_conf, sort_keys=True, default=str).encode()).hexdigest())
    save_dir = out_dir
    logger = yml_conf.get("logger", {})
    if logger.get("use_wandb"):
        save_dir = os.path.join(save_dir, logger["log_out_dir"])
    checkpoints = {}
    for directory in ("encoder", "full_model", "full_model_heir"):
        for path in sorted(glob(os.path.join(save_dir, directory, "**", "*"), recursive=True)):
            if os.path.isfile(path) and path.endswith((".ckpt", ".pkl", ".pt", ".pth")):
                stat = os.stat(path)
                checkpoints[path] = [stat.st_size, stat.st_mtime_ns]
    settings["checkpoints"] = checkpoints
    cache_matches = False
    if all(os.path.exists(path) for path in
           (explanation_path, values_path, indices_path, settings_path)):
        with open(settings_path) as f:
            cache_matches = json.load(f) == settings
        cache_matches = cache_matches and np.array_equal(np.load(indices_path), selected)
    if not overwrite and cache_matches:

        print("LOADING SHAP FILES")
        with open(explanation_path, "rb") as f:
            explanation = pickle.load(f)
        with open(values_path, "rb") as f:
            shap_values = pickle.load(f)

    else:
        # print("HERE", data.data_full.shape, data.data_full[0:10000].shape)
        # Compute with zero background
        # explanation = explain(get_output, data.data_full[0:10000], background, link=link,
        #                    output_names=label_names, out_dir=out_dir,
        #                    explanation_fname="explanation_zero_background.pkl")
        # shap_values = explanation.values
        # save_shap(shap_values, os.path.join(out_dir, "shap_values_zero_background.npz"))

        # Also compute with kmeans background
        # background = get_background('kmeans', data.data_full[0:10000], n_samples=1000)
        explanation = explain(get_output, dat, background, link=link,
                              output_names=label_names, out_dir=out_dir,
                              explanation_fname="explanation_kmeans_background.pkl",
                              nsamples=options["max_evals"] or "auto", rs=options["seed"])
        shap_values = explanation.values
        save_shap(shap_values, values_path)
        np.save(indices_path, selected)
        with open(settings_path, "w") as f:
            json.dump(settings, f)

    print("Shap values shape: ", np.array(shap_values).shape)
    print("HERE")
    print("Shap explanation shape: ", np.array(explanation).shape)
    print("HERE2")

    # ================================ PLOTTING ================================

    shap_out_dir = os.path.join(out_dir, "shap plots")
    os.makedirs(shap_out_dir, exist_ok=True)
    print(shap_out_dir)

    # feature_names = ["chan_" + str(i) for i in range(data.data_full.shape[1])]

    plot_by_freq = False  # True
    labels_by_freq = most_frequent_labels(output)
    print("Most significant labels: ", labels_by_freq)

    padding = yml_conf['data']['pixel_padding']
    tile_size = padding*2 + 1
    if tile_size > 1:
        dat = dat.reshape((shap_values.shape[0], int(
            shap_values.shape[1] / (tile_size**2)), tile_size, tile_size))
        dat = dat.reshape((dat.shape[0], dat.shape[1], -1))
        dat = np.moveaxis(dat, 1, 2)
        dat = dat.reshape((dat.shape[0]*dat.shape[1], dat.shape[2]))

        shap_values = shap_values.reshape((shap_values.shape[0], int(
            shap_values.shape[1] / (tile_size**2)), tile_size, tile_size))
        shap_values = shap_values.reshape(
            (shap_values.shape[0], shap_values.shape[1], -1))
        shap_values = np.moveaxis(shap_values, 1, 2)
        shap_values = shap_values.reshape(
            (shap_values.shape[0]*shap_values.shape[1], shap_values.shape[2]))
        output = np.repeat(output, tile_size**2, axis=0)

        # dat = np.max(dat, axis=(2,3)).reshape((shap_values.shape[0], shap_values.shape[1]))
        # shap_values = np.max(shap_values, axis=(2,3)).reshape((shap_values.shape[0], shap_values.shape[1]))

    feature_names = np.array(["chan_" + str(i)
                             for i in range(shap_values.shape[1])])

    print("HERE SHAP VALUES", shap_values.shape)
    if plot_by_freq:
        for label in labels_by_freq:
            inds = np.where(output == label)[0]
            if len(inds) < 1:
                continue

            save_summary_plot(
                out_dir, shap_values[inds, :], dat[inds, :], label, feature_names)
    else:
        for label in labels:
            inds = np.where(output == label)[0]
            if len(inds) < 1:
                continue

            save_summary_plot(
                out_dir, shap_values[inds, :], dat[inds, :], label, feature_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: remove nargs='?' from -y switch
    parser.add_argument("-y", "--yaml", nargs='?',
                        help="YAML file for cluster discretization.")
    parser.add_argument("-m", "--masker", nargs='?', help="Masker type.")
    parser.add_argument("-e", "--max-evals", type=int,
                        help="Coalition samples per SHAP/TimeSHAP explanation.")
    parser.add_argument("-b", "--batch-size", type=int,
                        help="Maximum inference batch size (default: 512).")
    parser.add_argument("--max-samples-per-cluster", type=int,
                        help="Spatial samples per predicted cluster (default: 200; 0: all).")
    parser.add_argument("--max-windows-per-cluster", type=int,
                        help="RTDBN windows per predicted cluster (default: 200; 0: all).")
    parser.add_argument("--seed", type=int, help="Sampling seed (default: 42).")
    args = parser.parse_args()
    from timeit import default_timer as timer
    start = timer()
    # TODO: update params
    main(**vars(args))
    end = timer()
    print(end - start)  # Time in seconds, e.g. 5.38091952400282

    # cleanup_ddp()


# def get_masker(masker, shape, fill):
#     """
#         args:
#             masker: string id of desired masker
#             shape: int tuple dimensions (bands, rows, cols) of data
#             fill: fill/no-data value for model
#     """

#     img_dims = shape
#     print("Image masker shape: ", img_dims)

#     if shape[0] == 1:
#         partition_scheme = 0
#     else:
#         partition_scheme = 1

#     print("Partition scheme: ", partition_scheme)

#     if masker == 'masker_inpaint_telea':
#         return shap.maskers.Image("inpaint_telea", img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_inpaint_ns':
#         return shap.maskers.Image("inpaint_ns", img_dims, partition_scheme=partition_scheme)
#     # TODO: Add support for maskers with custom blur kernel
#     if masker == 'masker_blur_3x3':
#         return shap.maskers.Image("blur(3, 3)", img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_blur_10x10':
#         return shap.maskers.Image("blur(10, 10)", img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_blur_100x100':
#         return shap.maskers.Image("blur(100, 100)", img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_uniform_black':
#         return shap.maskers.Image(np.zeros(img_dims), img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_uniform_gray':
#         return shap.maskers.Image(np.zeros(img_dims) + 128, img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_uniform_white':
#         return shap.maskers.Image(np.zeros(img_dims) + 255, img_dims, partition_scheme=partition_scheme)
#     if masker == 'uniform_fill':
#         return shap.maskers.Image(np.zeros(img_dims) + fill, img_dims, partition_scheme=partition_scheme)
