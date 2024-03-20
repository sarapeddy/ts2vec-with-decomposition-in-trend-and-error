import os
import time

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from utils_dlinear.tools_dlinear import EarlyStopping, adjust_learning_rate, visual, test_params_flop

from models.DLinear import Model
import torch
from tasks.forecasting import cal_metrics
from dataset_dlinear import DatasetDlinear


class DLinear:
    def __init__(self, device, n_time_cols, run_dir, name_dataset='ETTm1', seq_len=96, pred_len=24, enc_in=7, individual=False, batch_size=8, lr=0.0001, label_len=0, use_amp=False, output_attention=False,
                 features='M', num_workers=10, patience=3, train_epochs=10, test_flop=False, embed='timeF'):
        self.device = device
        self.n_time_cols = n_time_cols
        self.run_dir = run_dir
        self.name_dataset = name_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.label_len = label_len
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.patience = patience
        self.train_epochs = train_epochs
        self.use_amp = use_amp
        self.num_workers = num_workers
        self.output_attention = output_attention
        self.checkpoints = f'{self.run_dir}/checkpoints'
        self.test_flop = test_flop
        self.embed = embed
        self.model = Model(seq_len, pred_len, enc_in, individual).float().to(self.device)

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, train_data, vali_data, test_data, scaler):

        train_dataset = DatasetDlinear(torch.from_numpy(train_data).to(torch.float), seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len, n_time_cols=self.n_time_cols, flag='train')
        vali_dataset = DatasetDlinear(torch.from_numpy(vali_data).to(torch.float), seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len, n_time_cols=self.n_time_cols, flag='val')
        test_dataset = DatasetDlinear(torch.from_numpy(test_data).to(torch.float), seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len, n_time_cols=self.n_time_cols, flag='test')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        vali_loader = DataLoader(vali_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        path = os.path.join(self.checkpoints, 'DLinear')
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                        f_dim = -1 if self.features == 'MS' else 0
                        outputs = outputs[:, -self.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)

                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.features == 'MS' else 0
                    outputs = outputs[:, -self.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.lr)

            # break # Remove this line to train the model for the full number of epochs

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, test_data, scaler, results, test=0):
        test_dataset = DatasetDlinear(torch.from_numpy(test_data).to(torch.float), seq_len=self.seq_len,  label_len=self.label_len, pred_len=self.pred_len, n_time_cols=self.n_time_cols, flag='test')
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(f'{self.run_dir}/checkpoints/' + f'Dlinear__{self.seq_len}__{self.pred_len}', 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = f'{self.run_dir}/test_results/' + f'Dlinear__{self.seq_len}__{self.pred_len}' + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = f'{self.run_dir}/results/' + f'Dlinear__{self.seq_len}__{self.pred_len}' + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        results['ours'][self.pred_len] = {
            'norm': cal_metrics(preds.astype(np.float64), trues.astype(np.float64)),
            'raw': cal_metrics(scaler.inverse_transform(preds).astype(np.float64), scaler.inverse_transform(trues).astype(np.float64))
        }

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return results

    def predict(self, data, pred_slice, load=False):
        pred_dataset = DatasetDlinear(torch.from_numpy(data).to(torch.float), seq_len=self.seq_len, pred_len=self.pred_len, label_len=self.label_len, n_time_cols=self.n_time_cols, flag='pred')
        pred_loader = DataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False,drop_last=False)

        if load:
            path = os.path.join(self.checkpoints,  f'Dlinear__{self.seq_len}__{self.pred_len}')
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = f'{self.run_dir}/results/' +  f'Dlinear__{self.seq_len}__{self.pred_len}' + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
