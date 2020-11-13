import time
import datetime
import yaml
from pathlib import Path
from shutil import copyfile

from tqdm import tqdm
import torch
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from transformer import Transformer
from optimizer import CustomSchedule
from data import Dataset
from loss import vae_loss, ce_loss
from midi_io import MIDI


def timer(start, end):
    h, re = divmod(end - start, 3600)
    m, s = divmod(re, 60)
    return h, m, s


class Trainer:
    def __init__(self, model_name, style, resume):
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
        self.model_name = model_name
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.style = style
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resume = resume
        if resume:
            self.logdir = (Path.cwd() / "experiment" / "{}/{}/{}".format(model_name, style, resume[1])).as_posix()
            with (Path.cwd() / "experiment/{}/{}/{}/config.yaml".format(
                    model_name, style, resume[0]
            )).open(mode='r') as f:
                config = yaml.safe_load(f)
        else:
            self.logdir = (Path.cwd() / "experiment" / "{}/{}/{}".format(model_name, style, current_time)).as_posix()
            with (Path.cwd() / "model/transformer/config.yaml").open(mode='r') as f:
                config = yaml.safe_load(f)
        self.cfg = config[self.model_name]
        train_data = Dataset("train", style, "token", self.cfg[style]["seq_len"])
        self.train_loader = data.DataLoader(train_data,
                                            batch_size=self.cfg["batch_size"],
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=8)
        valid_data = Dataset("validation", style, "token", self.cfg[style]["seq_len"])
        self.valid_loader = data.DataLoader(valid_data,
                                            batch_size=4,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=8)

        self.model = Transformer(vocab=self.cfg[style]["vocab_size"],
                                 n_layer=self.cfg["n_layer"],
                                 n_head=self.cfg["n_head"],
                                 d_model=self.cfg["d_model"],
                                 d_head=self.cfg["d_head"],
                                 d_inner=self.cfg["d_inner"],
                                 dropout=self.cfg["dropout"]).to(self.device)

        if resume:
            self.model.load_state_dict(torch.load(
                Path.cwd() / "experiment/{}/{}/{}/model_{}.pt".format(
                    self.model_name, self.style, resume[0], resume[1])
            )["model_state_dict"])
        opt = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = CustomSchedule(self.cfg["d_model"], 1, 4000, optimizer=opt)
        if resume:
            self.scheduler.load_state_dict(torch.load(
                Path.cwd() / "experiment/{}/{}/{}/model_{}.pt".format(
                    self.model_name, self.style, resume[0], resume[1])
            )["optimizer_state_dict"])
        self.start_epoch = 1 if not resume else resume[1] + 1
        self.end_epoch = self.cfg["epoch"] + 1
        self.iteration = 0

    def train(self):
        writer = SummaryWriter(self.logdir)
        copyfile((Path.cwd() / "model/transformer/config.yaml").as_posix(), "{}/config.yaml".format(self.logdir))
        prev_valid_loss = 100000
        batch_size = self.cfg["batch_size"]

        for e in range(self.start_epoch, self.end_epoch):
            train_loss = 0
            train_acc = 0
            valid_loss = 0
            valid_acc = 0
            start_time = time.time()

            for i, batch in enumerate(self.train_loader):
                self.model.train()

                batch = batch.long().view(batch_size, -1).to(self.device)
                recon_batch = self.model.forward(batch[:, :-1])
                output_ce = 0
                output_kld = 0
                loss = ce_loss(recon_batch, batch[:, 1:])

                self.scheduler.optimizer.zero_grad()
                loss.backward()
                self.scheduler.step()
                norm = (batch != 0).sum().item()
                loss_norm = loss.item() / norm
                train_loss += loss_norm

                acc_norm = torch.eq(recon_batch.max(1)[1], batch[:, 1:]).sum().item() / norm * 100.

                train_acc += acc_norm
                # writer.add_scalar('TRAIN/ITER/LOSS', loss_norm, self.iteration)
                # writer.add_scalar('TRAIN/ITER/ACC', acc_norm, self.iteration)
                self.iteration += 1
                if i % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        e,
                        i * batch_size,
                        len(self.train_loader.dataset),
                        100. * i / len(self.train_loader)))

                    print('\t\t\tACC: {:.2f}\tLOSS: {:.6f}'.format(
                        acc_norm,
                        loss_norm))
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(self.valid_loader):

                    batch = batch.long().view(batch_size, -1).to(self.device)
                    recon_batch = self.model.forward(batch[:, :-1])
                    loss = ce_loss(recon_batch, batch[:, 1:])

                    norm = (batch != 0).sum().item()
                    loss_norm = loss.item() / norm
                    valid_loss += loss_norm

                    acc_norm = torch.eq(recon_batch.max(1)[1], batch[:, 1:]).sum().item() / norm * 100.

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
            print('\t\t\tTime taking: {:.0f}.{:.0f}.{:.0f}'.format(t[0], t[1], t[2]))
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

    def predict(self, model_name, style, length, durations, start_index=0):
        if style == "CPI":
            durations = False
        self.model.eval()
        num_excerpts = 30

        if style == "CSQ":
            z = Categorical(self.start_dist).sample(sample_shape=[num_excerpts, 1]).to(self.device)
        elif style == "CPI":
            z = torch.randint(low=357, high=389, size=(num_excerpts, 1)).to(self.device)
        else:
            raise ValueError("Invalid style.")

        with torch.no_grad():
            outputs = self.model.decode(z, length=length).cpu().numpy()
        for i, output in tqdm(enumerate(outputs)):
            mid = MIDI.to_midi(output.reshape(1, -1), durations)
            # mid = MIDI.to_midi(output, durations)
            mid.write((Path.cwd() / "experiment/{}/{}/{}/{}.mid".format(self.model_name,
                                                                        self.style,
                                                                        self.resume[0],
                                                                        start_index + i)).as_posix())

    def originality(self):
        writer = SummaryWriter(self.logdir)
        copyfile((Path.cwd() / "model/config.yaml").as_posix(), "{}/config.yaml".format(self.logdir))
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
                acc_norm = torch.eq(recon_batch.max(1)[1], batch[:, 1:]).sum().item() / norm * 100.
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
                            recon_batch = self.model.forward(valid_batch[:, :-1])
                            loss = ce_loss(recon_batch, valid_batch[:, 1:])
                            norm = (valid_batch != 0).sum().item()
                            loss_norm = loss.item() / norm
                            valid_loss += loss_norm
                            acc_norm = torch.eq(recon_batch.max(1)[1], valid_batch[:, 1:]).sum().item() / norm * 100.
                            valid_acc += acc_norm
                    writer.add_scalar('VALID/LOSS', valid_loss / len(self.valid_loader), self.iteration)
                    writer.add_scalar('VALID/ACC', valid_acc / len(self.valid_loader), self.iteration)
                    torch.save({'epoch': e,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.scheduler.state_dict()},
                               '{}/model_{}.pt'.format(self.logdir, self.iteration))
                    self.model.train()
                self.iteration += 1
            writer.flush()
