"""Temporal extension of SFDataset: slices data into overlapping, order-preserving windows."""
import os
import numpy as np
import torch
import pickle
from joblib import load, dump


class SFTemporalDataset(torch.utils.data.Dataset):
    """Dataset for sequence-structured (temporal) data, windowed over time."""

    def __init__(self):
        pass

    def init_from_array(self, data, targets, seq_len, scaler=None,
                        transform=None, subset_training=-1,
                        stratify_data=None, do_shuffle=True,
                        fill_value=-999999, scale=False,
                        train_scaler=False, scaler_out_path=None):
        """Initializes from a single continuous (N_timesteps, N_features) array."""
        self.data = data
        self.targets = targets
        self.seq_len = seq_len
        self.fill_value = fill_value
        self.scale = scale
        self.train_scaler = train_scaler
        self.scaler_out_path = scaler_out_path
        self.transform = transform
        self.subset_training = subset_training
        self.stratify_data = stratify_data
        self.do_shuffle = do_shuffle

        self.scaler = scaler

        if self.scale and self.train_scaler and self.scaler is not None:
            self.__fit_scaler__(data)
            if scaler_out_path is not None:
                with open(scaler_out_path, "wb") as f:
                    dump(self.scaler, f, True, pickle.HIGHEST_PROTOCOL)

        if self.scale and self.scaler is not None:
            data = self.__apply_scaler__(data)

        self.data = data
        self.__make_sequences__()

    def read_data_preprocessed(self, data_filename, indices_filename,
                               seq_len, scaler=None, subset_training=-1,
                               stratify_data=None, do_shuffle=True,
                               fill_value=-999999, scale=False,
                               train_scaler=False, scaler_out_path=None):
        """Loads pre-saved .npy data/index files and builds sequences."""
        data = np.load(data_filename, allow_pickle=True)
        targets = np.load(indices_filename, allow_pickle=True)
        self.init_from_array(
            data, targets, seq_len, scaler=scaler,
            subset_training=subset_training,
            stratify_data=stratify_data,
            do_shuffle=do_shuffle,
            fill_value=fill_value,
            scale=scale,
            train_scaler=train_scaler,
            scaler_out_path=scaler_out_path
        )

    def read_and_preprocess_data(self, filenames, seq_len, scaler=None,
                                 transform=None, subset_training=-1,
                                 stratify_data=None, do_shuffle=True,
                                 fill_value=-999999, scale=False,
                                 train_scaler=False, scaler_out_path=None):
        """Multi-trial entry point; windows each trial file separately, then combines."""
        self.filenames = filenames
        self.seq_len = seq_len
        self.fill_value = fill_value
        self.scale = scale
        self.train_scaler = train_scaler
        self.scaler_out_path = scaler_out_path
        self.transform = transform
        self.subset_training = subset_training
        self.stratify_data = stratify_data
        self.do_shuffle = do_shuffle

        if scaler is not None:
            self.scaler = scaler
        elif scale:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        all_trial_data = []
        for trial_idx, fname in enumerate(filenames):
            trial_data = np.load(fname, allow_pickle=True).astype(np.float32)
            n_timesteps = trial_data.shape[0]
            if n_timesteps < seq_len:
                print(
                    f"WARNING: trial {trial_idx} ({fname}) has only "
                    f"{n_timesteps} timesteps, less than seq_len="
                    f"{seq_len}. Skipping this trial entirely."
                )
                continue
            all_trial_data.append((trial_idx, trial_data))

        if len(all_trial_data) == 0:
            raise ValueError(
                f"No trial had enough timesteps to produce a single "
                f"sequence of length seq_len={seq_len}."
            )

        if self.scale and self.train_scaler and self.scaler is not None:
            combined = np.concatenate(
                [td for _, td in all_trial_data], axis=0
            )
            self.__fit_scaler__(combined)
            if scaler_out_path is not None:
                with open(scaler_out_path, "wb") as f:
                    dump(self.scaler, f, True, pickle.HIGHEST_PROTOCOL)

        all_sequences = []
        all_targets = []

        for trial_idx, trial_data in all_trial_data:
            if self.scale and self.scaler is not None:
                trial_data = self.__apply_scaler__(trial_data)

            trial_targets = np.array(
                [(trial_idx, t) for t in range(trial_data.shape[0])]
            )

            sequences, seq_targets = self.__window_single_trial__(
                trial_data, trial_targets, seq_len
            )
            all_sequences.append(sequences)
            all_targets.append(seq_targets)

        sequences = np.concatenate(all_sequences, axis=0)
        seq_targets = np.concatenate(all_targets, axis=0)

        sequences, seq_targets = self.__drop_fill_sequences__(
            sequences, seq_targets
        )

        if self.do_shuffle:
            order = np.random.permutation(len(sequences))
            sequences = sequences[order]
            seq_targets = seq_targets[order]

        sequences, seq_targets = self.__subset_sequences__(
            sequences, seq_targets
        )

        self.data_full = sequences
        self.targets_full = seq_targets

    def __fit_scaler__(self, data: np.ndarray) -> None:
        """Fits the scaler via partial_fit, excluding fill-value rows."""
        n_timesteps, n_features = data.shape
        flat = data.reshape(-1, n_features)
        valid_rows = flat[~np.any(flat == self.fill_value, axis=1)]
        if valid_rows.shape[0] > 0:
            self.scaler.partial_fit(valid_rows)

    def __apply_scaler__(self, data: np.ndarray) -> np.ndarray:
        """Applies the fitted scaler, preserving fill values."""
        n_timesteps, n_features = data.shape
        flat = data.reshape(-1, n_features)
        inds = np.where(flat == self.fill_value)
        flat = self.scaler.transform(flat)
        flat[inds] = self.fill_value
        return flat.reshape(n_timesteps, n_features)

    def __drop_fill_sequences__(self, sequences, seq_targets):
        """Drops any sequence where at least one frame has a fill value."""
        n_sequences = sequences.shape[0]
        flat = sequences.reshape(n_sequences, -1)
        del_inds = np.where(np.any(flat == self.fill_value, axis=1))[0]
        if len(del_inds) > 0:
            print(
                f"WARNING: dropping {len(del_inds)} sequences containing "
                f"fill value {self.fill_value}."
            )
            sequences = np.delete(sequences, del_inds, axis=0)
            seq_targets = np.delete(seq_targets, del_inds, axis=0)
        return sequences, seq_targets

    def __subset_sequences__(self, sequences, seq_targets):
        """Randomly subsets an already-built pool of sequences."""
        if self.subset_training is None or self.subset_training <= 0:
            return sequences, seq_targets

        if self.subset_training >= len(sequences):
            return sequences, seq_targets

        if self.stratify_data is not None and self.stratify_data.get("kmeans"):
            raise NotImplementedError(
                "K-means stratification is not yet implemented for "
                "sequence data."
            )

        indices = np.random.choice(
            len(sequences), size=self.subset_training, replace=False
        )
        return sequences[indices], seq_targets[indices]

    @staticmethod
    def __window_single_trial__(data, targets, seq_len):
        """Builds overlapping sequences via a step-1 sliding window over one trial."""
        n_timesteps, n_features = data.shape
        n_sequences = n_timesteps - seq_len + 1

        sequences = np.zeros((n_sequences, seq_len, n_features),
                             dtype=np.float32)
        seq_targets = []

        for i in range(n_sequences):
            sequences[i] = data[i:i + seq_len]
            seq_targets.append(targets[i + seq_len - 1])

        return sequences, np.array(seq_targets)

    def __make_sequences__(self):
        """Single-trial path used by init_from_array/read_data_preprocessed."""
        n_timesteps, n_features = self.data.shape

        if n_timesteps < self.seq_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) exceeds available timesteps "
                f"({n_timesteps})"
            )

        sequences, seq_targets = self.__window_single_trial__(
            self.data, self.targets, self.seq_len
        )

        sequences, seq_targets = self.__drop_fill_sequences__(
            sequences, seq_targets
        )

        if self.do_shuffle:
            order = np.random.permutation(len(sequences))
            sequences = sequences[order]
            seq_targets = seq_targets[order]

        sequences, seq_targets = self.__subset_sequences__(
            sequences, seq_targets
        )

        self.data_full = sequences
        self.targets_full = seq_targets

    def __len__(self):
        return len(self.data_full)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_full[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.targets_full[idx]
