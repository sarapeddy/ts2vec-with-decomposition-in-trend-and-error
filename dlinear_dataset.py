from torch.utils.data import Dataset


class DLinearDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_x = self.data[:, :-self.pred_len]
        self.data_y = self.data[:, -self.pred_len:]

    # Da rivedere per avere il giusto output quando si invoca il dataloader
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data)
