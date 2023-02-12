import time
import os
import errno
from typing import Optional
import datetime
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from shutil import copyfile

import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from models.vae import VAE
from models.transformer import Transformer
from models.optimizer import CustomSchedule
from models.data import Dataset
from models.loss import vae_loss, ce_loss, SmoothCrossEntropyLoss
from utilities.midi_io import MIDI
from models.data import CollateBatch

from models import (
    CSQ_START_DISTRIBUTION,
    CPI_START_DISTRIBUTION,
    CSQ_TIME_QUANTIZATION,
    CPI_TIME_QUANTIZATION
)


class TrainerBase:

    def __init__(
        self,
        device,
        cfg_path: Optional[str] = None,
        ckpt_path: Optional[str] = None
    ):
        if cfg_path is None and ckpt_path is None:
            raise ValueError(
                "Need either cfg_path (start training) or ckpt_path (resume training)."
            )
        elif cfg_path is not None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            self.cfg_path = cfg_path
            self.logdir = os.path.join("experiments", timestamp)
            copyfile(self.cfg_path, os.path.join(self.logdir, "config.yaml"))
        elif ckpt_path is not None:
            ckpt_dir = os.path.dirname(ckpt_path)
            self.cfg_path = os.path.join(ckpt_dir, "config.yaml")
            self.logdir = ckpt_dir
        
        self.writer = SummaryWriter(self.logdir)
        
        self.device = device
        with open(self.cfg_path) as f:
            self.cfg = load_hyperpyyaml(f)
        self.model = self.cfg["model"].to(self.device)

        train_data = Dataset(
            self.cfg["data_params"]["data_dir"],
            "train",
            self.cfg["data_params"]["seq_len"]
        )
        self.train_loader = data.DataLoader(
            train_data,
            batch_size=self.cfg["train_params"]["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=CollateBatch(device=self.device)
        )
        valid_data = Dataset(
            self.cfg["data_params"]["data_dir"],
            "validation",
            self.cfg["data_params"]["seq_len"]
        )
        self.valid_loader = data.DataLoader(
            valid_data,
            batch_size=self.cfg["train_params"]["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=CollateBatch(device=self.device)
        )
        opt = optim.Adam(
            self.model.parameters(),
            lr=0.00001,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.scheduler = CustomSchedule(
            self.cfg["d_model"],
            1,
            4000,
            optimizer=opt
        )
        self.start_epoch = 0
        self.end_epoch = self.cfg["train_params"]["epoch"]
        self.save_step = self.cfg["train_params"]["save_step"]

        if ckpt_path is not None:
            self.resume(ckpt_path)

    def resume(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.scheduler.load_state_dict(ckpt["optimizer_state_dict"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1

    def loss(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError


class MTTrainer(TrainerBase):

    def __init__(
        self,
        device,
        cfg_path: Optional[str] = None,
        ckpt_path: Optional[str] = None
    ):
        super().__init__(device, cfg_path, ckpt_path)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    def loss(self, batch, recon_batch):
        ce = self.loss_func(recon_batch, batch)
        return ce

    def train(self):
        for e in range(self.start_epoch, self.end_epoch):
            ##################
            #     Train      #
            ##################
            self.model.train()
            with tqdm(self.train_loader) as t:
                t.set_description(f'Epoch {e}')
                for i, batch in enumerate(t):
                    recon_batch = self.model(batch[:, :-1])
                    loss = self.loss(batch[:, 1:], recon_batch)

                    self.scheduler.optimizer.zero_grad()
                    loss.backward()
                    self.scheduler.step()

                    norm = (batch != 0).sum().item()
                    loss_norm = loss.item() / norm
                    acc_norm = torch.eq(
                        recon_batch.max(1)[1],
                        batch[:, 1:]
                    ).sum().item() / norm * 100.

                    self.writer.add_scalar('TRAIN/loss', loss_norm, e * len(self.train_loader) + i)
                    self.writer.add_scalar('TRAIN/accuracy', acc_norm, e * len(self.train_loader) + i)
                    t.set_postfix(loss=loss_norm, accuracy=acc_norm)

            ##################
            #   Validation   #
            ##################
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(self.valid_loader):
                    recon_batch = self.model(batch[:, :-1])
                    loss = self.loss(batch[:, 1:], recon_batch)

                    norm = (batch != 0).sum().item()
                    loss_norm = loss.item() / norm
                    acc_norm = torch.eq(
                        recon_batch.max(1)[1],
                        batch[:, 1:]
                    ).sum().item() / norm * 100.
                    self.writer.add_scalar('VALID/loss', loss_norm, e * len(self.train_loader) + i)
                    self.writer.add_scalar('VALID/accuracy', acc_norm, e * len(self.train_loader) + i)

            ##################
            #      Save      #
            ##################
            if e != 0 and e % self.save_step == 0:
                torch.save(
                    {
                        'epoch': e,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.scheduler.state_dict()
                    },
                    f'{self.logdir}/model_{e}.pt'
                )

    def generate(
        self,
        ckpt_path: str,
        num_excerpts: int,
        length: int,
        save_dir: str,
        time_quantization: list = CSQ_TIME_QUANTIZATION,
        time_unit: str = "crotchet",
        tempo: int = 120
    ):
        processor = MIDI(
            time_quantization=time_quantization,
            time_unit=time_unit
        )
        self.resume(ckpt_path)
        self.model.eval()
        with torch.no_grad():
            z = Categorical(
                torch.tensor(CSQ_START_DISTRIBUTION, device=self.device)
            ).sample(sample_shape=[num_excerpts, 1])
            outputs = self.model.decode(
                z, length=length).cpu().numpy()
        for i in range(num_excerpts):
            mid = processor.to_midi([outputs[i]], tempo=tempo)
            mid.write(os.path.join(save_dir, f'{i}.mid'))


class MVAETrainer(TrainerBase):

    def __init__(
        self,
        device,
        cfg_path: Optional[str] = None,
        ckpt_path: Optional[str] = None
    ):
        super().__init__(device, cfg_path, ckpt_path)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        self.kld_weight = 0

    def resume(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.scheduler.load_state_dict(ckpt["optimizer_state_dict"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.kld_weight = ckpt["kld_weight"]

    def loss(self, batch, recon_batch, mu, log_var):
        ce = self.loss_function(recon_batch, batch)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return ce, kld

    def train(self):
        for e in range(self.start_epoch, self.end_epoch):
            ##################
            #     Train      #
            ##################
            self.model.train()
            if e > self.cfg["warmup_epoch"] and self.kld_weight <= 0.2:
                self.kld_weight += self.cfg["warmup_rate"]
            with tqdm(self.train_loader) as t:
                t.set_description(f'Epoch {e}')
                for batch in t:
                    recon_batch = self.model(
                        batch.view(
                            batch.size(0),
                            self.model.num_tracks,
                            -1
                        )
                    )
                    ce, kld = self.loss(batch, recon_batch)
                    loss = ce + self.kld_weight * kld

                    self.scheduler.optimizer.zero_grad()
                    loss.backward()
                    self.scheduler.step()
                    breakpoint()
                    norm = (batch != 0).sum().item()
                    loss_norm = loss.item() / norm
                    acc_norm = torch.eq(
                        recon_batch.max(1)[1],
                        batch
                    ).sum().item() / norm * 100.

                    self.writer.add_scalar('TRAIN/loss', loss_norm)
                    self.writer.add_scalar('TRAIN/accuracy', acc_norm)
                    t.set_postfix(loss=loss_norm, accuracy=acc_norm)

            ##################
            #   Validation   #
            ##################
            self.model.eval()
            with torch.no_grad():
                for batch in self.valid_loader:
                    recon_batch = self.model(batch)
                    loss = self.loss(batch, recon_batch)

                    norm = (batch != 0).sum().item()
                    loss_norm = loss.item() / norm
                    acc_norm = torch.eq(
                        recon_batch.max(1)[1],
                        batch
                    ).sum().item() / norm * 100.
                    self.writer.add_scalar('VALID/loss', loss_norm)
                    self.writer.add_scalar('VALID/accuracy', acc_norm)

            ##################
            #      Save      #
            ##################
            if e != 0 and e % self.save_step == 0:
                torch.save(
                    {
                        'epoch': e,
                        'kld_weight': self.kld_weight,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.scheduler.state_dict()
                    },
                    f'{self.logdir}/model_{e}.pt'
                )

    def generate(
        self,
        ckpt_path: str,
        num_excerpts: int,
        length: int,
        save_dir: str,
        time_quantization: list,
        time_unit: str,
        tempo: int = 120
    ):
        processor = MIDI(
            time_quantization=time_quantization,
            time_unit=time_unit
        )
        self.resume(ckpt_path)
        self.model.eval()
        with torch.no_grad():
            z = torch.rand(num_excerpts, self.cfg['hidden_size'] * 2).to(self.device)
            outputs = self.model.decode(
                z, length=length).cpu().numpy()
        for i in range(num_excerpts):
            mid = processor.to_midi([outputs[i]], tempo=tempo)
            mid.write(os.path.join(save_dir, f'{i}.mid'))
