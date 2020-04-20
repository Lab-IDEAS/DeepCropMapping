import os
import sys
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.dcm import DCM
from config import SEED
from utils.timer import record_time


class DCMHelper(object):
    def __init__(
        self, seed=SEED, hidden_size=256, num_layers=2,
        bidirectional=True, dropout=0.5, batch_size=int(pow(2, 15)),
        num_workers=2, drop_last=False, criterion=nn.CrossEntropyLoss(),
        max_epoch=300, min_stop_epoch=5, stop_threshold=0.0002,
        record_attn=False, lr=0.0005, weight_decay=0.001, use_scheduler=True,
        mode="min", factor=0.5, patience=3, threshold=0.002,
        threshold_mode="abs", verbose=True
    ):
        # data description
        self.input_feature_size = None
        self.num_classes = None

        # model initialization (structure and hyperparameters)
        self.seed = seed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.dropout = dropout

        # data loader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        # training process
        self.criterion = criterion
        self.max_epoch = max_epoch
        self.min_stop_epoch = min_stop_epoch
        self.stop_threshold = stop_threshold
        self.record_attn = record_attn
        
        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        
        # scheduler
        self.use_scheduler = use_scheduler
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.verbose = verbose

        self.train_time_list = []
        self.test_time_list = []

    def _input_data(self, paths):
        data = np.concatenate(
            [np.load(path) for path in paths], axis=0
        )
        return data

    def input_x(self, paths):
        # x: (num_samples, seq_len, input_feature_size)
        x = self._input_data(paths).astype("float32")
        self.input_feature_size = x.shape[2]
        return x

    def input_y(self, paths):
        y = self._input_data(paths).astype("int64")
        self.num_classes = np.unique(y).shape[0]
        return y

    def normalize_with_scaler(self, scaler, x_test):
        x_test = scaler(
            torch.FloatTensor(x_test.transpose((0, 2, 1)))
        ).numpy().transpose((0, 2, 1))
        return x_test

    def normalize_without_scaler(self, x_train, x_test):
        scaler = torch.nn.BatchNorm1d(
            self.input_feature_size, eps=0, momentum=1, affine=False
        )
        scaler.train()
        x_train = scaler(
            torch.FloatTensor(x_train.transpose((0, 2, 1)))
        ).numpy().transpose((0, 2, 1))
        scaler.eval()
        x_test = scaler(
            torch.FloatTensor(x_test.transpose((0, 2, 1)))
        ).numpy().transpose((0, 2, 1))
        return scaler, x_train, x_test

    def _collate_fn(self, batch):
        """
        define how to aggregate samples to batch
        """
        return {
            "x": torch.FloatTensor(
                np.array([sample["sample_x"] for sample in batch])
            ),
            "y": torch.LongTensor(
                np.array([sample["sample_y"] for sample in batch])
            )
        }

    def make_data_loader(self, x, y, shuffle):
        return DataLoader(
            CropMappingDataset(x, y),
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, collate_fn=self._collate_fn,
            drop_last=self.drop_last
        )

    def build_model(self):
        return DCM(
            seed=self.seed,
            input_feature_size=self.input_feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            num_classes=self.num_classes
        )

    def _init_parameters(self, submodule):
        if type(submodule) == nn.LSTM:
            for name, param in submodule.named_parameters():
                if re.search("bias_ih", name):
                    # set forget gate bias to 3.0
                    param.detach().chunk(4)[1].fill_(3.0)

    def init_parameters(self, net):
        net.apply(self._init_parameters)

    def _eval_perf(self, net, dataloader, device):
        net.eval()
        with torch.no_grad():
            attn_batch_list = []
            losses = 0
            correct = 0
            for i, batch in enumerate(dataloader):
                xt, yt = batch["x"].to(device), batch["y"].to(device)
                outputs, attn_batch = net(xt)
                loss = self.criterion(outputs, yt)
                losses += loss.item() * yt.shape[0]
                yt_pred = torch.max(outputs, dim=1)[1]
                correct += (yt_pred == yt).sum().item()
                attn_batch_list.append(attn_batch)
            running_loss = losses / len(dataloader.dataset)
            acc = correct / len(dataloader.dataset)
            attn = torch.cat(attn_batch_list, dim=0).cpu().numpy()
        net.train()
        return running_loss, acc, attn

    def _train_model(
        self, net, train_dataloader, test_dataloader, device, logger,
        loss_train_list, acc_train_list, attn_train_list, loss_test_list,
        acc_test_list, attn_test_list
    ):
        optimizer = optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.mode, factor=self.factor,
            patience=self.patience, threshold=self.threshold,
            threshold_mode=self.threshold_mode, verbose=self.verbose
        )

        for epoch in range(1, self.max_epoch + 1):
            net.train()
            for i, batch in enumerate(train_dataloader):
                xt_train_batch = batch["x"].to(device)
                yt_train_batch = batch["y"].to(device)
                optimizer.zero_grad()
                outputs, _ = net(xt_train_batch)
                loss = self.criterion(outputs, yt_train_batch)
                loss.backward()
                optimizer.step()
            loss_train, acc_train, attn_train = self._eval_perf(
                net, train_dataloader, device
            )
            if self.use_scheduler:
                scheduler.step(loss_train)  # adjust learning rate
            loss_test, acc_test, attn_test = self._eval_perf(
                net, test_dataloader, device
            )
            loss_train_list.append(loss_train)
            acc_train_list.append(acc_train)
            loss_test_list.append(loss_test)
            acc_test_list.append(acc_test)
            if self.record_attn:
                attn_train_list.append(attn_train)
                attn_test_list.append(attn_test)

            logger.info((
                "[epoch {:d}] "
                "training loss: {:.4f}, test loss: {:.4f}, "
                "training acc: {:.4f}, test acc: {:.4f} "
                " (lr => {:f})"
            ).format(
                epoch,
                loss_train, loss_test,
                acc_train, acc_test,
                optimizer.param_groups[0]["lr"])
            )

            if (
                epoch >= self.min_stop_epoch
                and (
                    np.array(acc_train_list[epoch - 5:epoch]).ptp()
                    <= self.stop_threshold
                )
            ):
                break

        logger.info("Training completed")

    def train_model(
        self, net, train_dataloader, test_dataloader, device, logger,
        loss_train_list, acc_train_list, attn_train_list, loss_test_list,
        acc_test_list, attn_test_list
    ):
        record_time(self.train_time_list, self._train_model, [
            net, train_dataloader, test_dataloader, device, logger,
            loss_train_list, acc_train_list, attn_train_list, loss_test_list,
            acc_test_list, attn_test_list
        ])

    def _predict(self, net, dataloader, device):
        yt_soft_pred_batch_list = []
        attn_batch_list = []
        net.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                xt, yt = batch["x"].to(device), batch["y"].to(device)
                outputs, attn_batch = net(xt)
                yt_soft_pred_batch_list.append(F.softmax(outputs, dim=1))
                attn_batch_list.append(attn_batch)
            y_soft_pred = torch.cat(
                yt_soft_pred_batch_list, dim=0
            ).cpu().numpy()
            y_hard_pred = np.argmax(y_soft_pred, axis=1)
            attn = torch.cat(attn_batch_list, dim=0).cpu().numpy()
        return y_soft_pred, y_hard_pred, attn

    def predict(self, net, dataloader, device):
        return record_time(self.test_time_list, self._predict, [
            net, dataloader, device
        ])


class CropMappingDataset(Dataset):
    """
    crop classification dataset
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {"sample_x": self.x[idx], "sample_y": self.y[idx]}
