import time
import os
import errno
import json

import datetime
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from shutil import copyfile

import numpy as np

from tqdm import tqdm
import torch
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

from models import (
    CSQ_START_DISTRIBUTION,
    CPI_START_DISTRIBUTION,
    CSQ_TIME_QUANTIZATION,
    CPI_TIME_QUANTIZATION
)


def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def timer(start, end):
    h, re = divmod(end - start, 3600)
    m, s = divmod(re, 60)
    return h, m, s


class CollateBatch:

    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        return torch.from_numpy(np.stack(batch, axis=0)).long().to(self.device)


class TrainerBase:

    def __init__(
        self,
        cfg_path,
        device
    ):
        self.device = device
        with open("hyperparameters.yaml") as f:
            cfg = load_hyperpyyaml(f)
        self.model = cfg["model"].to(self.device)
        train_data = Dataset(
            cfg["data_params"]["data_dir"],
            "train",
            cfg["data_params"]["seq_len"]
        )
        self.train_loader = data.DataLoader(
            train_data,
            batch_size=cfg["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=CollateBatch(device=self.device)
        )
        valid_data = Dataset(
            cfg["data_params"]["data_dir"],
            "validation",
            cfg["data_params"]["seq_len"]
        )
        self.valid_loader = data.DataLoader(
            valid_data,
            batch_size=cfg["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=CollateBatch(device=self.device)
        )
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.logdir = os.path.join("experiments", timestamp)
        self.writer = SummaryWriter(self.logdir)
        copyfile(cfg_path, os.path.join(self.logdir, "config.yaml"))
        opt = optim.Adam(
            self.model.parameters(),
            lr=0.00001,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.scheduler = CustomSchedule(
            self.model.d_model,
            1,
            4000,
            optimizer=opt
        )
        self.sce_loss = SmoothCrossEntropyLoss(
            0.1,
            self.model.vocab_size
        )
        self.start_epoch = 0
        self.end_epoch = cfg["train_params"]["epoch"]
        self.save_step = cfg["train_params"]["save_step"]

    def resume(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.scheduler.load_state_dict(ckpt["optimizer_state_dict"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1

    def loss(self, batch):
        raise NotImplementedError

    def train(self, ckpt_path: str):
        raise NotImplementedError

    def generate(self, ckpt_path: str):
        raise NotImplementedError


class MTTrainer(TrainerBase):

    def __init__(self, cfg_path, device):
        super().__init__(cfg_path, device)

    def loss(self, batch, recon_batch):
        return self.sce_loss(recon_batch.transpose(1, 2), batch[:, 1:])

    def train(self, ckpt_path: str = None):
        if ckpt_path is not None:
            self.resume(ckpt_path)

        for e in range(self.start_epoch, self.end_epoch):
            ##################
            #     Train      #
            ##################
            self.model.train()
            with tqdm(self.train_loader) as t:
                t.set_description(f'Epoch {e}')
                for batch in t:
                    recon_batch = self.model(batch[:, :-1])
                    loss = self.loss(batch, recon_batch)

                    self.scheduler.optimizer.zero_grad()
                    loss.backward()
                    self.scheduler.step()

                    norm = (batch != 0).sum().item()
                    loss_norm = loss.item() / norm
                    acc_norm = torch.eq(
                        recon_batch.max(1)[1],
                        batch[:, 1:]
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
                    recon_batch = self.model(batch[:, :-1])
                    loss = self.loss(batch, recon_batch)

                    norm = (batch != 0).sum().item()
                    loss_norm = loss.item() / norm
                    acc_norm = torch.eq(
                        recon_batch.max(1)[1],
                        batch[:, 1:]
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
            z = Categorical(
                torch.tensor(CSQ_START_DISTRIBUTION, device=self.device)
                ).sample(sample_shape=[num_excerpts, 1])
            outputs = self.model.decode(z, length=length).unsqueeze(1).cpu().numpy()
        for i in range(num_excerpts):
            mid = processor.to_midi([outputs[i]], tempo=tempo)
            mid.write(os.path.join(save_dir, f'{i}.mid'))


class Trainer(TrainerBase):
    def __init__(self, model_name, style, resume):
        if style == 'CSQ':
            self.start_dist = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00024224806,
                 0.00024224806, 0.00024224806, 0.00048449612, 0.0007267442, 0.0012112403, 0.0012112403,
                 0.0016957364, 0.0019379845, 0.0024224806, 0.003633721, 0.004118217, 0.0058139535, 0.00629845,
                 0.0067829457, 0.007025194, 0.007751938, 0.009689922, 0.010416667, 0.013081395, 0.013323643,
                 0.01501938, 0.017199613, 0.019137597, 0.019137597, 0.020833334, 0.026405038, 0.027858527,
                 0.030281007, 0.033430234, 0.034399226, 0.034883723, 0.035610463, 0.038032945, 0.037790697,
                 0.037790697, 0.039244186, 0.03754845, 0.03585271, 0.034641474, 0.033430234, 0.029554263,
                 0.029796511, 0.028343024, 0.02761628, 0.02374031, 0.021075582, 0.019137597, 0.016472869,
                 0.016472869, 0.013323643, 0.0125969, 0.010658915, 0.008963178, 0.007267442, 0.0053294576,
                 0.004844961, 0.004118217, 0.003633721, 0.003149225, 0.0024224806, 0.0019379845, 0.0007267442,
                 0.0007267442, 0.00048449612, 0.00048449612, 0.00024224806, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0]
            )
        elif style == 'CPI':
            self.start_dist = torch.tensor(
                [0.0013410404624277456, 0.0019421965317919076, 0.0024508670520231213, 0.003144508670520231,
                 0.004439306358381503, 0.005734104046242775, 0.011930635838150289, 0.016138728323699423,
                 0.02423121387283237, 0.03408092485549133, 0.04171098265895954, 0.050589595375722544, 0.059838150289017344,
                 0.06113294797687861, 0.0691329479768786, 0.07213872832369943, 0.07764161849710982, 0.07796531791907514,
                 0.07986127167630058, 0.07773410404624277, 0.06363005780346821, 0.0524393063583815, 0.04240462427745665,
                 0.029919075144508672, 0.021040462427745665, 0.010589595375722544, 0.004763005780346821,
                 0.0013410404624277456, 0.0005086705202312139, 0.00018497109826589596, 0.0, 0.0]
            )

        self.model_name = model_name
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.style = style
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.resume = resume
        if resume:
            self.logdir = f"./experiment/{model_name}/{style}/{resume[0]}"
            with open(f"./experiment/{model_name}/{style}/{resume[0]}/config.yaml", 'r') as f:
                config = yaml.safe_load(f)
        else:
            self.logdir = f'./experiment/{model_name}/{style}/{current_time}'
            with open("./model/config.yaml", 'r') as f:
                config = yaml.safe_load(f)
        self.cfg = config[self.model_name]
        train_data = Dataset("train", style, "token",
                             self.cfg[style]["seq_len"])
        self.train_loader = data.DataLoader(train_data,
                                            batch_size=self.cfg["batch_size"],
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0)
        valid_data = Dataset("validation", style, "token",
                             self.cfg[style]["seq_len"])
        self.valid_loader = data.DataLoader(valid_data,
                                            batch_size=self.cfg["batch_size"],
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0)
        if self.model_name == "VAE":
            self.model = VAE(input_size=self.cfg["input_size"],
                             hidden_size=self.cfg["hidden_size"],
                             encoder_num_layers=self.cfg["encoder_num_layers"],
                             conductor_num_layers=self.cfg["conductor_num_layers"],
                             decoder_num_layers=self.cfg["decoder_num_layers"],
                             dropout=self.cfg["dropout"],
                             vocab_size=self.cfg[style]["vocab_size"],
                             num_tracks=self.cfg[style]["num_tracks"]).to(self.device)
        elif self.model_name == "Transformer":
            self.model = Transformer(vocab=self.cfg[style]["vocab_size"],
                                     n_layer=self.cfg["n_layer"],
                                     n_head=self.cfg["n_head"],
                                     d_model=self.cfg["d_model"],
                                     d_head=self.cfg["d_head"],
                                     d_inner=self.cfg["d_inner"],
                                     dropout=self.cfg["dropout"]).to(self.device)

        if resume:
            self.model.load_state_dict(torch.load(
                f"./experiment/{self.model_name}/{self.style}/{resume[0]}/model_{resume[1]}.pt"
            )["model_state_dict"])
        opt = optim.Adam(self.model.parameters(), lr=0.0001,
                         betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = CustomSchedule(
            self.cfg["d_model"], 1, 4000, optimizer=opt)
        if resume:
            self.scheduler.load_state_dict(torch.load(
                f"./experiment/{self.model_name}/{self.style}/{resume[0]}/model_{resume[1]}.pt"
            )["optimizer_state_dict"])
        self.start_epoch = 1 if not resume else resume[1] + 1
        self.end_epoch = self.cfg["epoch"] + 1
        self.iteration = 0
        self.sce_loss = SmoothCrossEntropyLoss(
            0.1, self.cfg[style]["vocab_size"])

    def train(self):
        writer = SummaryWriter(self.logdir)
        copyfile("./model/config.yaml", f"{self.logdir}/config.yaml")
        prev_valid_loss = 100000
        batch_size = self.cfg["batch_size"]

        kld_weight = 0
        warmup_epoch = self.cfg["warmup_epoch"] if self.model_name == "VAE" else 0
        warmup_rate = self.cfg["warmup_rate"] if self.model_name == "VAE" else 0

        for e in range(self.start_epoch, self.end_epoch):
            train_loss = 0
            train_acc = 0
            valid_loss = 0
            valid_acc = 0
            start_time = time.time()
            if e > warmup_epoch and kld_weight <= 0.2:
                kld_weight += warmup_rate
            for i, batch in enumerate(self.train_loader):
                self.model.train()
                if self.model_name == "VAE":
                    batch = batch.long().view(
                        batch_size, self.cfg[self.style]["num_tracks"], -1).to(self.device)
                    recon_batch, output_mu, output_log_var = self.model.forward(
                        batch)
                    output_ce, output_kld = vae_loss(recon_batch,
                                                     batch,
                                                     output_mu,
                                                     output_log_var)
                    loss = output_ce + kld_weight * output_kld
                elif self.model_name == "Transformer":
                    batch = batch.long().view(batch_size, -1).to(self.device)
                    recon_batch = self.model.forward(batch[:, :-1])
                    output_ce = 0
                    output_kld = 0
                    loss = self.sce_loss(
                        recon_batch.transpose(1, 2), batch[:, 1:])
                else:
                    raise ValueError("Invalid model name.")

                self.scheduler.optimizer.zero_grad()
                loss.backward()
                self.scheduler.step()
                norm = (batch != 0).sum().item()
                loss_norm = loss.item() / norm
                train_loss += loss_norm
                if self.model_name == "VAE":
                    acc_norm = torch.eq(recon_batch.max(
                        1)[1], batch).sum().item() / norm * 100.
                elif self.model_name == "Transformer":
                    acc_norm = torch.eq(recon_batch.max(
                        1)[1], batch[:, 1:]).sum().item() / norm * 100.
                else:
                    raise ValueError("Invalid model name.")
                train_acc += acc_norm
                writer.add_scalar('TRAIN/ITER/LOSS', loss_norm, self.iteration)
                writer.add_scalar('TRAIN/ITER/ACC', acc_norm, self.iteration)
                self.iteration += 1
                if (e == 1) and (i in [100, 300, 1000, 5000]):
                    torch.save({'epoch': e,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.scheduler.state_dict()},
                               f'{self.logdir}/model_{e}-{i}.pt')

                if i % 10 == 0:
                    print(f'Train Epoch: {e} [{i * batch_size}/{len(self.train_loader.dataset)} ' +
                          f'({100. * i / len(self.train_loader):.0f}%)]')

                    if self.model_name == "VAE":
                        print(f'\t\t\tACC: {acc_norm:.2f}\tLOSS: {loss_norm:.6f}' +
                              f'\tCE: {output_ce.item() / norm:.6f}\tKLD: {output_kld.item() / norm:.6f}')
                    elif self.model_name == "Transformer":
                        print(
                            f'\t\t\tACC: {acc_norm:.2f}\tLOSS: {loss_norm:.6f}')
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(self.valid_loader):
                    if self.model_name == "VAE":
                        batch = batch.long().view(
                            batch_size, self.cfg[self.style]["num_tracks"], -1).to(self.device)
                        recon_batch, output_mu, output_log_var = self.model.forward(
                            batch)
                        output_ce, output_kld = vae_loss(recon_batch,
                                                         batch,
                                                         output_mu,
                                                         output_log_var)
                        loss = output_ce + kld_weight * output_kld
                    elif self.model_name == "Transformer":
                        batch = batch.long().view(batch_size, -1).to(self.device)
                        recon_batch = self.model.forward(batch[:, :-1])
                        loss = self.sce_loss(
                            recon_batch.transpose(1, 2), batch[:, 1:])
                    else:
                        raise ValueError("Invalid model name.")
                    norm = (batch != 0).sum().item()
                    loss_norm = loss.item() / norm
                    valid_loss += loss_norm
                    if self.model_name == "VAE":
                        acc_norm = torch.eq(recon_batch.max(
                            1)[1], batch).sum().item() / norm * 100.
                    elif self.model_name == "Transformer":
                        acc_norm = torch.eq(recon_batch.max(
                            1)[1], batch[:, 1:]).sum().item() / norm * 100.
                    else:
                        raise ValueError("Invalid model name.")
                    valid_acc += acc_norm

            average_train_loss = train_loss / len(self.train_loader)
            average_valid_loss = valid_loss / len(self.valid_loader)
            average_train_acc = train_acc / len(self.train_loader)
            average_valid_acc = valid_acc / len(self.valid_loader)
            t = timer(start_time, time.time())
            print('Epoch: {}\tAverage train Loss: {:.4f}\tAccuracy: {:.2f}'.format(
                e, average_train_loss, average_train_acc))
            print('\t\t\tAverage validation Loss {:.4f}\tAccuracy: {:.2f}'.format(
                average_valid_loss, average_valid_acc))
            print('\t\t\tTime taking: {:.0f}.{:.0f}.{:.0f}'.format(
                t[0], t[1], t[2]))
            writer.add_scalar('TRAIN/LOSS', average_train_loss, e)
            writer.add_scalar('VALID/LOSS', average_valid_loss, e)
            writer.add_scalar('TRAIN/ACC', average_train_acc, e)
            writer.add_scalar('VALID/ACC', average_valid_acc, e)
            writer.flush()
            torch.save({'epoch': e,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.scheduler.state_dict()},
                       '{}/model_{}.pt'.format(self.logdir, e))
            if average_valid_loss < prev_valid_loss:
                torch.save({'epoch': e,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.scheduler.state_dict()},
                           '{}/model_best.pt'.format(self.logdir))
                prev_valid_loss = average_valid_loss

    # def predict(self, model_name, style, length, durations, start_index=0):
    #     if style == "CPI":
    #         durations = False
    #     self.model.eval()
    #     num_excerpts = 50
    #
    #     if style == "CSQ":
    #         z = Categorical(self.csq_start_dist).sample(sample_shape=[num_excerpts, 1]).to(self.device)
    #     elif style == "CPI":
    #         z = (Categorical(self.cpi_start_dist).sample(sample_shape=[num_excerpts, 1]) + 357).to(self.device)
    #         # z = torch.randint(low=357, high=389, size=(num_excerpts, 1)).to(self.device)
    #     else:
    #         raise ValueError("Invalid style.")
    #
    #     with torch.no_grad():
    #         outputs = self.model.decode(z, length=length).cpu().numpy()
    #     for i, output in tqdm(enumerate(outputs)):
    #         mid = MIDI.to_midi(output.reshape(1, -1), durations)
    #         # mid = MIDI.to_midi(output, durations)
    #         filename = f"./experiment/{self.model_name}/{self.style}/" + \
    #                    f"{self.resume[0]}/{self.resume[1]}/{start_index + i}"
    #         mkdir(filename)
    #         with open(filename + ".json", 'w') as outfile:
    #             json.dump(output.tolist(), outfile)
    #         mid.write(filename + ".mid")

    def predict(self, model_name, style, length, durations, start_index=0):
        if style == "CPI":
            durations = False
        self.model.eval()
        num_excerpts = 2
        if model_name == "VAE":
            z = torch.rand(
                num_excerpts, self.cfg['hidden_size'] * 2).to(self.device)
        elif model_name == "Transformer":
            z = Categorical(self.start_dist).sample(
                sample_shape=[num_excerpts, 1]).to(self.device)
        else:
            raise ValueError("Invalid model name.")

        with torch.no_grad():
            outputs = self.model.decode(z, length=length).cpu().numpy()
        for i, output in tqdm(enumerate(outputs)):
            mid = MusicBox.to_midi(output.reshape(1, -1), durations)
            # mid = MIDI.to_midi(output, durations)
            mid.write((Path.cwd() / "experiment/{}/{}/{}/{}.mid".format(self.model_name,
                                                                        self.style,
                                                                        self.resume[0],
                                                                        start_index + i)).as_posix())

    def originality(self):
        writer = SummaryWriter(self.logdir)
        copyfile((Path.cwd() / "model/config.yaml").as_posix(),
                 "{}/config.yaml".format(self.logdir))
        batch_size = self.cfg["batch_size"]
        self.model.train()
        for e in range(self.start_epoch, self.end_epoch):
            for i, batch in enumerate(self.train_loader):
                batch = batch.long().view(batch_size, -1).to(self.device)
                recon_batch = self.model.forward(batch[:, :-1])
                loss = ce_loss(recon_batch, batch[:, 1:])
                self.scheduler.optimizer.zero_grad()
                loss.backward()
                self.scheduler.step()
                norm = (batch != 0).sum().item()
                loss_norm = loss.item() / norm
                acc_norm = torch.eq(recon_batch.max(
                    1)[1], batch[:, 1:]).sum().item() / norm * 100.
                writer.add_scalar('TRAIN/LOSS', loss_norm, self.iteration)
                writer.add_scalar('TRAIN/ACC', acc_norm, self.iteration)
                if i % 11 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        e,
                        i * batch_size,
                        len(self.train_loader.dataset),
                        100. * i / len(self.train_loader)))
                    print('\t\t\tACC: {:.2f}\tLOSS: {:.6f}'.format(
                        acc_norm,
                        loss_norm))
                if i % 500 == 0:
                    self.model.eval()
                    valid_loss = 0
                    valid_acc = 0
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        e,
                        i * batch_size,
                        len(self.train_loader.dataset),
                        100. * i / len(self.train_loader)))
                    print('\t\t\tACC: {:.2f}\tLOSS: {:.6f}'.format(
                        acc_norm,
                        loss_norm))
                    with torch.no_grad():
                        for valid_batch in tqdm(self.valid_loader):
                            valid_batch = valid_batch.long().view(batch_size, -1).to(self.device)
                            recon_batch = self.model.forward(
                                valid_batch[:, :-1])
                            loss = ce_loss(recon_batch, valid_batch[:, 1:])
                            norm = (valid_batch != 0).sum().item()
                            loss_norm = loss.item() / norm
                            valid_loss += loss_norm
                            acc_norm = torch.eq(recon_batch.max(
                                1)[1], valid_batch[:, 1:]).sum().item() / norm * 100.
                            valid_acc += acc_norm
                    writer.add_scalar(
                        'VALID/LOSS', valid_loss / len(self.valid_loader), self.iteration)
                    writer.add_scalar('VALID/ACC', valid_acc /
                                      len(self.valid_loader), self.iteration)
                    torch.save({'epoch': e,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.scheduler.state_dict()},
                               '{}/model_{}.pt'.format(self.logdir, self.iteration))
                    self.model.train()
                self.iteration += 1
            writer.flush()
