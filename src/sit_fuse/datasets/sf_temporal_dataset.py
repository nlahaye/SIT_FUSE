"""
Temporal extension of SFDataset for sequence/time-series data.
Modeled on sit_fuse.datasets.sf_dataset.SFDataset's vector-based path
(init_from_array / read_data_preprocessed), with spatial windowing
replaced by causal temporal windowing.
"""
import numpy as np
import torch


class SFTemporalDataset(torch.utils.data.Dataset):
    """
    Dataset class for sequence-structured (temporal) data, e.g. biomechanical
    time series. Extension of SFDataset's vector-based ingestion path:
    instead of treating each row as an independent unordered sample, this
    class slices the data into overlapping, ORDER-PRESERVING sequences of a
    fixed length using a sliding window over the time axis.
    """

    def __init__(self):
        pass

    def init_from_array(self, data, targets, seq_len, scaler=None,
                         transform=None, subset_training=-1,
                         stratify_data=None, do_shuffle=True):
        self.data = data
        self.targets = targets
        self.seq_len = seq_len
        self.scaler = scaler
        self.scale = scaler is not None
        self.transform = transform
        self.subset_training = subset_training
        self.stratify_data = stratify_data
        self.do_shuffle = do_shuffle

        self.__make_sequences__()

    def read_data_preprocessed(self, data_filename, indices_filename,
                                seq_len, scaler=None, subset_training=-1,
                                stratify_data=None, do_shuffle=True):
        data = np.load(data_filename, allow_pickle=True)
        targets = np.load(indices_filename, allow_pickle=True)
        self.init_from_array(data, targets, seq_len, scaler=scaler,
                              subset_training=subset_training,
                              stratify_data=stratify_data,
                              do_shuffle=do_shuffle)

    def read_and_preprocess_data(self, filenames, seq_len, scaler=None,
                                  transform=None, subset_training=-1,
                                  stratify_data=None, do_shuffle=True):
        """
        Multi-trial entry point. Mirrors SFDataset.read_and_preprocess_data's
        `filenames`-list convention (sf_dataset.py): each file is one
        continuous trial/recording, shape (n_timesteps_i, n_features).
        Trials may have DIFFERENT lengths (e.g. a short reach vs. a long
        walk) -- that's fine, each is windowed independently.

        Sequences are built PER TRIAL FIRST (so a sequence never straddles
        two trials/recordings), and only the resulting SEQUENCE LISTS are
        concatenated across trials -- never the raw per-trial time series.

        :param filenames: list of paths to .npy files, each one trial.
        :param seq_len: sequence length, applied identically to every trial.
        :param scaler: optional per-feature scaler.
        :param transform: optional transform, applied per sequence.
        :param subset_training: kept for interface parity; not yet ported.
        :param stratify_data: kept for interface parity; not yet ported.
        :param do_shuffle: if True, shuffles the COMBINED list of
            sequences across ALL trials. Never within a sequence, never
            across trial boundaries.
        """
        self.filenames = filenames
        self.seq_len = seq_len
        self.scaler = scaler
        self.scale = scaler is not None
        self.transform = transform
        self.subset_training = subset_training
        self.stratify_data = stratify_data
        self.do_shuffle = do_shuffle

        all_sequences = []
        all_targets = []

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

            trial_targets = np.array(
                [(trial_idx, t) for t in range(n_timesteps)]
            )

            sequences, seq_targets = self.__window_single_trial__(
                trial_data, trial_targets, seq_len
            )
            all_sequences.append(sequences)
            all_targets.append(seq_targets)

        if len(all_sequences) == 0:
            raise ValueError(
                "No trial had enough timesteps to produce a single "
                f"sequence of length seq_len={seq_len}."
            )

        sequences = np.concatenate(all_sequences, axis=0)
        seq_targets = np.concatenate(all_targets, axis=0)

        if self.do_shuffle:
            order = np.random.permutation(len(sequences))
            sequences = sequences[order]
            seq_targets = seq_targets[order]

        sequences, seq_targets = self.__subset_sequences__(sequences, seq_targets)

        self.data_full = sequences
        self.targets_full = seq_targets

    def __subset_sequences__(self, sequences, seq_targets):
        """
        Internal. Applies subset_training / stratify_data to an already-
        built pool of sequences, shared by both the single-trial and
        multi-trial paths.

        RESOLVED (per Nick, meeting 2026-06-29): plain random subsetting
        is unambiguous for sequences -- "take N random sequences" means
        the same thing whether a sample is a pixel or a sequence -- so
        it's implemented below.

        STILL DEFERRED: sf_dataset.py's k-means-based stratification
        (__stratify_k_means__) is NOT ported. It's genuinely unclear what
        "a representative subset" should mean for sequences -- cluster on
        the raw flattened sequence? On some summary statistic? Per-trial
        or pooled across all trials? Raises NotImplementedError if
        requested, rather than guessing at a definition.
        """
        if self.subset_training is None or self.subset_training <= 0:
            return sequences, seq_targets

        if self.subset_training >= len(sequences):
            # Nothing to subset -- requested size already covers (or
            # exceeds) what we have.
            return sequences, seq_targets

        if self.stratify_data is not None and self.stratify_data.get("kmeans"):
            raise NotImplementedError(
                "K-means-based stratification is not yet defined for "
                "sequence data (unclear what 'representative subset' "
                "means here -- cluster on raw sequence? summary stats? "
                "per-trial? pooled?). Confirm approach with Nick before "
                "implementing. Plain random subset_training (without "
                "stratify_data) is supported."
            )

        indices = np.random.choice(
            len(sequences), size=self.subset_training, replace=False
        )
        return sequences[indices], seq_targets[indices]

    @staticmethod
    def __window_single_trial__(data, targets, seq_len):
        """
        Internal. Builds overlapping, causal sequences via a sliding window
        over the time axis (step=1) for ONE trial, never crossing into
        another trial. Shared by both the single-array path
        (__make_sequences__) and the multi-trial path
        (read_and_preprocess_data) so they can't drift apart.
        """
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
        """
        Internal. Single-trial path used by init_from_array /
        read_data_preprocessed. Delegates to __window_single_trial__.
        """
        n_timesteps, n_features = self.data.shape

        if n_timesteps < self.seq_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) exceeds available timesteps "
                f"({n_timesteps})"
            )

        sequences, seq_targets = self.__window_single_trial__(
            self.data, self.targets, self.seq_len
        )

        if self.do_shuffle:
            order = np.random.permutation(len(sequences))
            sequences = sequences[order]
            seq_targets = seq_targets[order]

        sequences, seq_targets = self.__subset_sequences__(sequences, seq_targets)

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