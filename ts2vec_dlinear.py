import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

def create_batch_ci(x):
    x = x.unsqueeze(3)
    x = x.reshape(x.shape[0] * x.shape[2], x.shape[1], x.shape[3])
    return x

def create_batch_inv_ci(x, batch_size, feature, dims):
    x = x.reshape(batch_size, feature, dims)
    return x

class TS2VecDlinear:
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
        self._net_err = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net_avg = torch.optim.swa_utils.AveragedModel(self._net_avg)
        self.net_err = torch.optim.swa_utils.AveragedModel(self._net_err)
        self.net_avg.update_parameters(self._net_avg)
        self.net_err.update_parameters(self._net_err)

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
        optimizer2 = torch.optim.AdamW(self._net_err.parameters(), lr=self.lr)

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
                    x = x[:, window_offset : window_offset + self.max_train_length]
                    y = y[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                y = y.to(self.device)

                if self.ci:
                    batch_size, feature, dims = x.shape
                    assert torch.equal(x, create_batch_inv_ci(create_batch_ci(x), batch_size, feature, dims))
                    assert torch.equal(y, create_batch_inv_ci(create_batch_ci(y), batch_size, feature, dims))
                    x = create_batch_ci(x)
                    y = create_batch_ci(y)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                # First model: average
                out1_avg = self._net_avg(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1_avg = out1_avg[:, -crop_l:]

                out2_avg = self._net_avg(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2_avg = out2_avg[:, :crop_l]

                # Second model; error
                out1_err = self._net_err(take_per_row(y, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1_err = out1_err[:, -crop_l:]

                out2_err = self._net_err(take_per_row(y, crop_offset + crop_left, crop_eright - crop_left))
                out2_err = out2_err[:, :crop_l]

                if self.mode == 'ts2vec-Dlinear-two-loss':
                    loss1 = hierarchical_contrastive_loss(
                        out1_avg,
                        out2_avg,
                        temporal_unit=self.temporal_unit
                    )

                    loss2 = hierarchical_contrastive_loss(
                        out1_err,
                        out2_err,
                        temporal_unit=self.temporal_unit
                    )
                    loss = loss1 + loss2
                else:
                    loss = hierarchical_contrastive_loss(
                        out1_avg + out1_err,
                        out2_avg + out2_err,
                        temporal_unit=self.temporal_unit
                    )

                loss.backward()
                optimizer1.step()
                optimizer2.step()
                self.net_avg.update_parameters(self._net_avg)
                self.net_err.update_parameters(self._net_err)

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
    
    def _eval_with_pooling(self, x, y, mask=None, slicing=None, encoding_window=None):
        out1 = self.net_err(x.to(self.device, non_blocking=True), mask)
        out2 = self.net_avg(y.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out1 = out1[:, slicing]
                out2 = out2[:, slicing]
            out1 = F.max_pool1d(
                out1.transpose(1, 2),
                kernel_size = out1.size(1),
            ).transpose(1, 2)
            out2 = F.max_pool1d(
                out2.transpose(1, 2),
                kernel_size=out2.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out1 = F.max_pool1d(
                out1.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            out2 = F.max_pool1d(
                out2.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out1 = out1[:, :-1]
                out2 = out2[:, :-1]
            if slicing is not None:
                out1 = out1[:, slicing]
                out2 = out2[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs1 = []
            reprs2 = []
            while (1 << p) + 1 < out1.size(1):
                t_out1 = F.max_pool1d(
                    out1.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                t_out2 = F.max_pool1d(
                    out2.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out1 = t_out1[:, slicing]
                    t_out2 = t_out2[:, slicing]
                reprs1.append(t_out1)
                reprs2.append(t_out2)
                p += 1
            out1 = torch.cat(reprs1, dim=-1)
            out2 = torch.cat(reprs2, dim=-1)
            
        else:
            if slicing is not None:
                out1 = out1[:, slicing]
                out2 = out2[:, slicing]
            
        return out1.cpu(), out2.cpu()
    
    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
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
        assert self.net_err is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training_avg = self.net_avg.training
        org_training_err = self.net_avg.training
        self.net_avg.eval()
        self.net_err.eval()

        dataset = TimeSeriesDatasetWithMovingAvg(torch.from_numpy(data).to(torch.float), self.n_time_cols)
        loader = create_custom_dataLoader(dataset, batch_size, n_time_cols=self.n_time_cols, eval=True)
        
        with torch.no_grad():
            output1 = []
            output2 = []
            for x, y in loader:
                if sliding_length is not None:
                    reprs1 = []
                    reprs2 = []
                    if n_samples < batch_size:
                        calc_buffer1 = []
                        calc_buffer2 = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        print(i)
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        y_sliding = torch_pad_nan(
                            y[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out1, out2 = self._eval_with_pooling(
                                    torch.cat(calc_buffer1, dim=0),
                                    torch.cat(calc_buffer2, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs1 += torch.split(out1, n_samples)
                                reprs2 += torch.split(out2, n_samples)
                                calc_buffer1 = []
                                calc_buffer2 = []
                                calc_buffer_l = 0
                            calc_buffer1.append(x_sliding)
                            calc_buffer2.append(y_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out1, out2 = self._eval_with_pooling(
                                x_sliding,
                                y_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs1.append(out1)
                            reprs2.append(out2)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out1, out2 = self._eval_with_pooling(
                                torch.cat(calc_buffer1, dim=0),
                                torch.cat(calc_buffer2, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs1 += torch.split(out1, n_samples)
                            reprs2 += torch.split(out2, n_samples)
                            calc_buffer1 = []
                            calc_buffer2 = []
                            calc_buffer_l = 0
                    
                    out1 = torch.cat(reprs1, dim=1)
                    out2 = torch.cat(reprs2, dim=1)
                    if encoding_window == 'full_series':
                        out1 = F.max_pool1d(
                            out1.transpose(1, 2).contiguous(),
                            kernel_size = out1.size(1),
                        ).squeeze(1)
                        out2 = F.max_pool1d(
                            out2.transpose(1, 2).contiguous(),
                            kernel_size=out2.size(1),
                        ).squeeze(1)
                else:
                    out1, out2 = self._eval_with_pooling(x, y, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out1 = out1.squeeze(1)
                        out2 = out2.squeeze(1)
                        
                output1.append(out1)
                output2.append(out2)

                # break # only one iteration

            output1  = torch.cat(output1, dim=0)
            output2  = torch.cat(output2, dim=0)

        output = output1 + output2
        self.net_avg.train(org_training_avg)
        self.net_err.train(org_training_err)
        return output.numpy()

    def _eval_with_pooling_ci(self, x, y, mask=None, slicing=None, encoding_window=None):
        batch_size, _, feature = x.shape
        x = create_batch_ci(x)
        y = create_batch_ci(y)

        out1 = self.net_err(x.to(self.device, non_blocking=True), mask)
        out2 = self.net_avg(y.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out1 = out1[:, slicing]
                out2 = out2[:, slicing]
            out1 = F.max_pool1d(
                out1.transpose(1, 2),
                kernel_size=out1.size(1),
            ).transpose(1, 2)
            out2 = F.max_pool1d(
                out2.transpose(1, 2),
                kernel_size=out2.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out1 = F.max_pool1d(
                out1.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            out2 = F.max_pool1d(
                out2.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out1 = out1[:, :-1]
                out2 = out2[:, :-1]
            if slicing is not None:
                out1 = out1[:, slicing]
                out2 = out2[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            reprs1 = []
            reprs2 = []
            while (1 << p) + 1 < out1.size(1):
                t_out1 = F.max_pool1d(
                    out1.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                t_out2 = F.max_pool1d(
                    out2.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out1 = t_out1[:, slicing]
                    t_out2 = t_out2[:, slicing]
                reprs1.append(t_out1)
                reprs2.append(t_out2)
                p += 1
            out1 = torch.cat(reprs1, dim=-1)
            out2 = torch.cat(reprs2, dim=-1)

        else:
            if slicing is not None:
                out1 = out1[:, slicing]
                out2 = out2[:, slicing]
                out1 = create_batch_inv_ci(out1, batch_size, feature, self.output_dims)
                out2 = create_batch_inv_ci(out2, batch_size, feature, self.output_dims)

        return out1.cpu(), out2.cpu()

    def encode_ci(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0,
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
        assert self.net_err is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training_avg = self.net_avg.training
        org_training_err = self.net_avg.training
        self.net_avg.eval()
        self.net_err.eval()

        # dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        dataset = TimeSeriesDatasetWithMovingAvg(torch.from_numpy(data).to(torch.float), self.n_time_cols)
        loader = create_custom_dataLoader(dataset, batch_size, n_time_cols=self.n_time_cols, eval=True)

        with torch.no_grad():
            output1 = []
            output2 = []
            for x, y in loader:
                if sliding_length is not None:
                    reprs1 = []
                    reprs2 = []
                    if n_samples < batch_size:
                        calc_buffer1 = []
                        calc_buffer2 = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):

                        if i % 1000 == 0:
                            print(f'Processing {i} timestamps')

                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        y_sliding = torch_pad_nan(
                            y[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out1, out2 = self._eval_with_pooling_ci(
                                    torch.cat(calc_buffer1, dim=0),
                                    torch.cat(calc_buffer2, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs1 += torch.split(out1, n_samples)
                                reprs2 += torch.split(out2, n_samples)
                                calc_buffer1 = []
                                calc_buffer2 = []
                                calc_buffer_l = 0
                            calc_buffer1.append(x_sliding)
                            calc_buffer2.append(y_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out1, out2 = self._eval_with_pooling_ci(
                                x_sliding,
                                y_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs1.append(out1)
                            reprs2.append(out2)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out1, out2 = self._eval_with_pooling_ci(
                                torch.cat(calc_buffer1, dim=0),
                                torch.cat(calc_buffer2, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs1 += torch.split(out1, n_samples)
                            reprs2 += torch.split(out2, n_samples)
                            calc_buffer1 = []
                            calc_buffer2 = []
                            calc_buffer_l = 0

                    out1 = torch.cat(reprs1, dim=0)
                    out2 = torch.cat(reprs2, dim=0)

                    if encoding_window == 'full_series':
                        out1 = F.max_pool1d(
                            out1.transpose(1, 2).contiguous(),
                            kernel_size=out1.size(1),
                        ).squeeze(1)
                        out2 = F.max_pool1d(
                            out2.transpose(1, 2).contiguous(),
                            kernel_size=out2.size(1),
                        ).squeeze(1)
                else:
                    out1, out2 = self._eval_with_pooling_ci(x, y, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out1 = out1.squeeze(1)
                        out2 = out2.squeeze(1)

                output1.append(out1.unsqueeze(0))
                output2.append(out2.unsqueeze(0))

                # break # only one iteration

            output1 = torch.cat(output1, dim=0)
            output2 = torch.cat(output2, dim=0)

        output = output1 + output2
        self.net_avg.train(org_training_avg)
        self.net_err.train(org_training_err)
        return output.numpy()

    def save(self, fn1, fn2):
        ''' Save the model to a file.
        
        Args:
            fn1 (str): filename.
            fn2 (str): filename.
        '''
        torch.save(self.net_avg.state_dict(), fn1)
        torch.save(self.net_err.state_dict(), fn2)

    def load(self, fn1, fn2):
        ''' Load the model from a file.
        
        Args:
            fn1 (str): filename.
            fn2 (str): filename.
        '''
        state_dict_avg = torch.load(fn1, map_location=self.device)
        state_dict_err = torch.load(fn2, map_location=self.device)
        self.net_avg.load_state_dict(state_dict_avg)
        self.net_err.load_state_dict(state_dict_err)

