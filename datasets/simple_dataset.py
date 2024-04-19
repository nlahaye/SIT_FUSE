

class SimpleDataset(Dataset):
    def __init__(self,
                 data,
                 ):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class SimpleDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size=16,
                 num_workers=4,
                 pin_memory=True,
                 shuffle=True,
                 val_percent=0.1
                 ):
        super().__init__()

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, stage=None):
        data = np.load(dataset_path)
        train_max = int(data.shape[0]*(1.0-val_percent)
        print(train_max, data.shape)
        self.train_dataset = SimpleDataset(torch.from_numpy(data[:train_max]))
        self.val_dataset = SimpleDataset(torch.from_numpy(data[train_max:]))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


