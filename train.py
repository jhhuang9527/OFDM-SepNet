import os, sys
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import Trainer, seed_everything
from asteroid.engine import System
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pickle
from pytorch_lightning.loggers import CSVLogger
import argparse
import torch.nn.functional as F

from nets.waveunet import Waveunet
from nets.OFDM_SepNet import OFDM_SepNet
from nets.SepMamba import SepMambaCausal

np.random.seed(42)
random.seed(42)
seed_everything(42, workers=True)

""" running parameters"""
parser = argparse.ArgumentParser(
    description="Train parameters for running SepNet."
)

parser.add_argument("--file_root", type=str, default='/your_root_path/')
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--model", type=str, default='SepMamba') # waveunet SepMamba OFDM_SepNet
parser.add_argument("--min_snr", type=float, default=0.0, help="Minimum SNR in dB for training")
parser.add_argument("--max_snr", type=float, default=10.0, help="Maximum SNR in dB for training")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--lr_milestones", nargs='+', type=int, default=[80, 160], help="Epochs at which to reduce LR")
parser.add_argument("--lr_gamma", type=float, default=0.1, help="Factor by which to reduce LR at milestones")


args = parser.parse_args()

sig_len = 512

def add_awgn_noise_torch(signals: torch.Tensor, snr_db: float) -> torch.Tensor:
    signals = signals.float()
    signal_power = 1.0
    
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    sigma = torch.sqrt(torch.tensor(noise_power, device=signals.device, dtype=signals.dtype))
    noise = sigma * torch.randn_like(signals)

    return signals + noise

class SNRAgnosticDataset(Dataset):
    def __init__(self, all_signals_mix_clean, target_first_clean_signal, min_snr, max_snr):
        self.all_signals_mix_clean = torch.from_numpy(all_signals_mix_clean).float()
        self.target_first_clean_signal = torch.from_numpy(target_first_clean_signal).float()
        self.min_snr = min_snr
        self.max_snr = max_snr

        assert len(self.all_signals_mix_clean) == len(self.target_first_clean_signal), \
            "All original signals and target clean signal must have the same number of samples."

    def __len__(self):
        return len(self.all_signals_mix_clean)

    def __getitem__(self, idx):
        current_sample_all_clean_signals = self.all_signals_mix_clean[idx] # (6, 512)

        y_clean = current_sample_all_clean_signals[0:1, :] # (1, 512)

        random_snr_db = random.uniform(self.min_snr, self.max_snr)
        snr_tensor = torch.tensor([random_snr_db], dtype=torch.float32) # (scalar)

        y_noisy_original = add_awgn_noise_torch(y_clean, random_snr_db) # (1, 512)

        x_input_signals_modified = current_sample_all_clean_signals.clone() # (6, 512)
        x_input_signals_modified[0:1, :] = y_noisy_original 
        
        x_model_input = torch.sum(x_input_signals_modified, dim=0, keepdim=True) # (1, 512)
        
        return x_model_input, y_clean, y_noisy_original, snr_tensor


all_train_mix, all_train_clean, _ = pickle.load(open(os.path.join(f'{args.file_root}/datasets', f'train/train.pickle'),'rb'))
n_train = int(len(all_train_mix)*0.8)

train_all_sig_mix_clean = all_train_mix[:n_train,:,:] # (N_train, 6, L)
train_all_sig_clean_first = all_train_clean[:n_train,:,:] # (N_train, 1, L)

val_all_sig_mix_clean = all_train_mix[n_train:,:,:] # (N_val, 6, L)
val_all_sig_clean_first = all_train_clean[n_train:,:,:] # (N_val, 1, L)
    
train_dataset = SNRAgnosticDataset(
    train_all_sig_mix_clean,
    train_all_sig_clean_first,
    args.min_snr,
    args.max_snr
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    pin_memory=True
)

val_dataset = SNRAgnosticDataset(
    val_all_sig_mix_clean,
    val_all_sig_clean_first,
    args.min_snr,
    args.max_snr
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    pin_memory=True
)

class MyCustomSystem(System):
    def __init__(self, model, optimizer, train_loader, val_loader, lr_milestones, lr_gamma):
        super().__init__(model, optimizer, None, train_loader, val_loader)
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma

    def common_step(self, batch, batch_idx):
        x_model_input, y_clean_label, y_noisy_original_label, snr_value = batch
        
        y_hat = self.model(x_model_input, snr_value) # y_hat: (B, 2, L)

        pred_clean = y_hat[:, 0:1, :] # (B, 1, L)
        pred_noisy_original = y_hat[:, 1:2, :] # (B, 1, L)

        loss_clean = F.mse_loss(pred_clean, y_clean_label)
        loss_noisy_original = F.mse_loss(pred_noisy_original, y_noisy_original_label)

        total_loss = loss_clean + loss_noisy_original
        
        return total_loss, loss_clean, loss_noisy_original

    def training_step(self, batch, batch_idx):
        total_loss, loss_clean, loss_noisy_original = self.common_step(batch, batch_idx)

        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_clean", loss_clean, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_loss_noisy_original", loss_noisy_original, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, loss_clean, loss_noisy_original = self.common_step(batch, batch_idx)

        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_clean", loss_clean, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss_noisy_original", loss_noisy_original, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma),
            'interval': 'epoch',
            'frequency': 1,
        }
        
        return [optimizer], [scheduler]

def train_script(idx):
    if idx == 'waveunet':
        model = Waveunet(n_src=2, n_first_filter=20, depth=5)
    elif idx == 'SepMamba':
        model = SepMambaCausal(dim=64)
    elif idx == 'OFDM_SepNet':
        model = OFDM_SepNet()
    else:
        raise ValueError(f"Unknown model type: {idx}")

    model.cuda()
    
    log_name = f"{args.model}_SNRs"
    tb_logger = CSVLogger(
        save_dir=f"{args.file_root}/logs",
        name=f"{args.model}",
        version='v0',
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    system = MyCustomSystem(
        model,
        optimizer,
        train_loader,
        val_loader,
        lr_milestones=args.lr_milestones,
        lr_gamma=args.lr_gamma
    )
    
    ckpt_dir = os.path.join(f'{args.file_root}/checkpoints', f"{args.model}/{log_name}_models")

    ckpt_best = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor='val_total_loss',
        mode='min',
        filename='{epoch}-{val_total_loss:.4f}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=tb_logger,
        devices="auto",
        callbacks=[ckpt_best, lr_monitor]
    )
    trainer.fit(system)

if __name__ == '__main__':   
    train_script(args.model)
    print('done')

