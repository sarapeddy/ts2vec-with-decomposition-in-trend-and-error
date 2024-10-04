import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss
from moving_avg_tensor_dataset import TimeSeriesDatasetWithMovingAvg
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan


def custom_collate_fn(batch, n_time_cols=7):
    # Stack della lista di tensori in un unico tensore
    data = torch.stack([item[0] for item in batch], dim=0)
    total_covariate = (data.shape[2] - n_time_cols)//2

    result_data_avg = torch.cat([data[:, :, :n_time_cols], data[:, :, n_time_cols:n_time_cols+total_covariate]], dim=2)
    result_data_err = torch.cat([data[:, :, :n_time_cols], data[:, :, n_time_cols + total_covariate:]], dim=2)
    return result_data_avg, result_data_err

def create_custom_dataLoader(dataset, batch_size, n_time_cols=7, eval=False):
    def collate_fn(batch):
        return custom_collate_fn(batch, n_time_cols=n_time_cols)

    if eval:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=True, collate_fn=collate_fn)

def transform_ci(x, B, F, T):
    x = torch.swapaxes(x, 1, 2)
    x = x.reshape(B * F, T, 1)
    return x

def transform_inv_ci(x, B, F, T, E):
    x = x.reshape(B, F, T, E)
    x = torch.swapaxes(x, 1, 2)
    x = x.reshape(B, T, F * E)
    return x

class TS2VecAblation:
    '''The TS2Vec model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
        mode='ts2vec-Dlinear-two-loss',
        n_time_cols=0,
        ci=False
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.output_dims = output_dims
        
        self._net_avg = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        # self._net_err = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net_avg = torch.optim.swa_utils.AveragedModel(self._net_avg)
        # self.net_err = torch.optim.swa_utils.AveragedModel(self._net_err)
        self.net_avg.update_parameters(self._net_avg)
        # self.net_err.update_parameters(self._net_err)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
        self.mode = mode
        self.n_time_cols = n_time_cols
        self.ci = ci
    
    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        train_dataset = TimeSeriesDatasetWithMovingAvg(torch.from_numpy(train_data).to(torch.float), n_time_cols=self.n_time_cols)
        train_loader = create_custom_dataLoader(train_dataset, self.batch_size, n_time_cols=self.n_time_cols)
        
        optimizer1 = torch.optim.AdamW(self._net_avg.parameters(), lr=self.lr)
        # optimizer2 = torch.optim.AdamW(self._net_err.parameters(), lr=self.lr)

        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for x, y in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                if self.max_train_length is not None and x.size(1) > self.max_train_length and y.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    # x = x[:, window_offset : window_offset + self.max_train_length]
                    y = y[:, window_offset : window_offset + self.max_train_length]

                # x = x[:, :, self.n_time_cols:]
                # y = y[:, :, self.n_time_cols:]

                # x = x.to(self.device)
                # y = y.to(self.device)
                x = y.to(self.device)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer1.zero_grad()
                # optimizer2.zero_grad()

                B, T, F = x.shape
                _, T_avg1, _ = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft).shape
                _, T_avg2, _ = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left).shape
                # _, T_err1, _ = take_per_row(y, crop_offset + crop_eleft, crop_right - crop_eleft).shape
                # _, T_err2, _ = take_per_row(y, crop_offset + crop_left, crop_eright - crop_left).shape

                if self.ci:
                    x_spt1 = transform_ci(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft), B, F, T_avg1)
                    x_spt2 = transform_ci(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left), B, F, T_avg2)
                    # y_spt1 = transform_ci(take_per_row(y, crop_offset + crop_eleft, crop_right - crop_eleft), B, F, T_err1)
                    # y_spt2 = transform_ci(take_per_row(y, crop_offset + crop_left, crop_eright - crop_left), B, F, T_err2)
                else:
                    x_spt1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
                    x_spt2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
                    # y_spt1 = take_per_row(y, crop_offset + crop_eleft, crop_right - crop_eleft)
                    # y_spt2 = take_per_row(y, crop_offset + crop_left, crop_eright - crop_left)

                # First model: average
                out1_avg = self._net_avg(x_spt1)
                out1_avg = transform_inv_ci(out1_avg, B, F, T_avg1, self.output_dims) if self.ci else out1_avg
                out1_avg = out1_avg[:, -crop_l:]

                out2_avg = self._net_avg(x_spt2)
                out2_avg = transform_inv_ci(out2_avg, B, F, T_avg2, self.output_dims) if self.ci else out2_avg
                out2_avg = out2_avg[:, :crop_l]

                # Second model; error
                # out1_err = self._net_err(y_spt1)
                # out1_err = transform_inv_ci(out1_err, B, F, T_err1, self.output_dims) if self.ci else out1_err
                # out1_err = out1_err[:, -crop_l:]
                #
                # out2_err = self._net_err(y_spt2)
                # out2_err = transform_inv_ci(out2_err, B, F, T_err2, self.output_dims) if self.ci else out2_err
                # out2_err = out2_err[:, :crop_l]

                loss = hierarchical_contrastive_loss(
                    out1_avg,
                    out2_avg,
                    temporal_unit=self.temporal_unit
                )

                loss.backward()
                optimizer1.step()
                # optimizer2.step()
                self.net_avg.update_parameters(self._net_avg)
                # self.net_err.update_parameters(self._net_err)

                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

                # break # only one iteration

            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

            # break # only one epoch

        return loss_log

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        # x = x[:, :, self.n_time_cols:]

        B, T, Fe = x.shape

        if self.ci:
            x = transform_ci(x, B, Fe, T)

        out = self.net_avg(x.to(self.device, non_blocking=True), mask)

        if self.ci:
            out = transform_inv_ci(out, B, Fe, T, self.output_dims)

        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0,
               batch_size=None):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        '''
        assert self.net_avg is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net_avg.training
        self.net_avg.eval()

        dataset = TimeSeriesDatasetWithMovingAvg(torch.from_numpy(data).to(torch.float), self.n_time_cols)
        loader = create_custom_dataLoader(dataset, batch_size, n_time_cols=self.n_time_cols, eval=True)

        with torch.no_grad():
            output = []
            for x, y in loader:
                x = y
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net_avg.train(org_training)
        return output.numpy()

    def save(self, fn):
        ''' Save the model to a file.

        Args:
            fn (str): filename.
        '''
        torch.save(self.net_avg.state_dict(), fn)

    def load(self, fn):
        ''' Load the model from a file.

        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net_avg.load_state_dict(state_dict)

