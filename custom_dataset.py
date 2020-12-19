import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, path, seq_len, use_cols, testratio=1.0):
        self.files = glob.glob(os.path.join(path, '*'))
        self.usecols = use_cols
        self.seq_len = seq_len
        self.data = []
        self.labels = []
        for i in range(len(self.files)):
            datum, label = self.processing(i)
            if self.data:
                self.data.extend(datum)
                self.labels.extend(label)
                assert len(self.data) == len(self.labels)
            else:
                self.data = datum
                self.labels = label
        if testratio < 1.0:
            num_samples = int(len(self.data) * testratio)
            self.data = self.data[:num_samples]
            self.labels = self.labels[:num_samples]

        self.data = torch.from_numpy(np.asarray(self.data))
        self.labels = torch.from_numpy(np.asarray(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def processing(self, idx):
        datum = []
        label = []
        df = pd.read_csv(self.files[idx], usecols=self.usecols)
        df = df.sort_values(by=['Timestamp'], axis=0)
        df.loc[df.IOType == 'R', 'IOType'] = 0.0
        df.loc[df.IOType == 'W', 'IOType'] = 1.0
        df['Idletime'] = df.Timestamp - df.Timestamp.shift(1)
        df.Idletime = df.Idletime.fillna(0)
        df.drop(['Timestamp'], axis=1, inplace=True)
        idletimes = np.array(df['Idletime'], dtype=np.float32)
        values = np.array(df, dtype=np.float32)
        for i in range(len(values) - self.seq_len + 1):
            last_index = i + self.seq_len - 1
            datum.append(values[i:last_index])
            label.append(idletimes[last_index])
        return datum, label



