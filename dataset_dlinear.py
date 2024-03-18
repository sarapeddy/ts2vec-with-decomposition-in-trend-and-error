from torch.utils.data import Dataset


class DatasetDlinear(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, n_time_cols, flag='train', target='OT', cols=None, inverse=False):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.n_time_cols = n_time_cols
        self.target = target
        self.flag = flag
        self.cols = cols
        self.inverse = inverse
        self.data = self.data.squeeze(0)

        if self.flag == 'pred':
            if self.cols:
                cols = self.cols.copy()
                cols.remove(self.target)
            else:
                cols = list(self.data[:, self.n_time_cols:].columns)
                cols.remove(self.target)
            self.data = self.data[:, self.n_time_cols:].values

        self.data_x = self.data[:, self.n_time_cols:]
        self.data_y = self.data[:, self.n_time_cols:]

    # Da rivedere per avere il giusto output quando si invoca il dataloader
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.flag == 'pred':
            seq_x = self.data_x[s_begin:s_end]
            if self.inverse:
                seq_y = self.data_x[r_begin:r_begin + self.label_len]
            else:
                seq_y = self.data_y[r_begin:r_begin + self.label_len]
            return seq_x, seq_y

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        if self.flag == 'pred':
            return len(self.data_x) - self.seq_len + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1