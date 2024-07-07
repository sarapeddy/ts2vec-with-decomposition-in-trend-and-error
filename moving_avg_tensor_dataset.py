import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class TimeSeriesDatasetWithMovingAvg(TensorDataset):
    
    def __init__(self, original_dataset: Tensor, n_time_cols,  kernel_size=9):
        self.n_time_cols = n_time_cols
        self.moving_avg = MovingAvg(kernel_size, stride=1)
        x_time = original_dataset[:, :, :self.n_time_cols]
        x_original = original_dataset[:, :, self.n_time_cols:]
        x_avg = self.moving_avg(x_original)
        x_err = x_original - x_avg
        expanded_dataset = torch.cat([x_time, x_avg, x_err], dim=2)
        super(TimeSeriesDatasetWithMovingAvg, self).__init__(expanded_dataset)

