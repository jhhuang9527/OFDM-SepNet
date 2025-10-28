
import argparse
import numpy as np
import random
from pytorch_lightning import Trainer, seed_everything
import torch
from torch.utils.data import TensorDataset, DataLoader
import glob
import os
from torch import optim
from asteroid.engine import System
import math
import pandas as pd
from tqdm import tqdm


from nets.waveunet import Waveunet
from nets.OFDM_SepNet import OFDM_SepNet
from nets.SepMamba import SepMambaCausal



""" running parameters"""
parser = argparse.ArgumentParser(
    description="Parameters for seprator."
)
parser.add_argument("--file_root", type=str, default='/your_root_path/')
parser.add_argument("--num_user", type=int, default=2)
parser.add_argument("--num_ofdm", type=int, default=50)
parser.add_argument("--coeff_step", type=float, default=0.5)
parser.add_argument("--model", type=str, default='waveunet') # waveunet SepMamba OFDM_SepNet
parser.add_argument("--snr", type=float, default=5.0)
parser.add_argument("--num_sample", type=int, default=1000)

args = parser.parse_args()

seed_everything(420123, workers=True)
np.random.seed(420123)
random.seed(420123)

def add_awgn_noise(signals, snr_db, signal_power=1.0):
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power)
    noise = sigma * np.random.randn(*signals.shape)
    
    return signals + noise

def gen_ofdm(coeff=1, nfft=64, n_sc=29, osfactor=1, sig_len=512):
    np.random.seed(0)
    random.seed(0)
    cos_waves = np.exp(1j*2*np.pi*osfactor*np.arange(nfft).reshape(-1,1)/nfft*(np.arange(sig_len).reshape(1,-1)))
    syms = (2*np.random.randint(2, size=(nfft, 1)) - 1)
    syms[0,:] = 0
    syms[n_sc+1:,:] = 0
    syms[nfft//2+1:,:] = np.flipud(syms[1:nfft//2,:])
    sig_comp = syms * coeff * 1/np.sqrt(2*n_sc) * cos_waves
    sig = sig_comp.sum(axis=0)

    return sig, syms[1:n_sc+1,:]

def gen_data_multi(user_num, ofdm_num, power):
    sigs, syms = [], []
    for i in range(user_num):
        sig, sym = [], []
        power_i = power[i]
        for _ in range(ofdm_num):
            ofdm_sig, ofdm_sym = gen_ofdm(coeff=power_i)
            sig.extend(ofdm_sig)
            sym.extend(ofdm_sym)
        sigs.append(sig)
        syms.append(sym)

    return sigs, syms

def add_with_offset(arr1, arr2, k):
    arr1 = arr1.flatten()
    arr2 = arr2.flatten()
    n, m = len(arr1), len(arr2)

    result = arr1[:k].tolist()

    overlap_len = min(n - k, m)
    for i in range(overlap_len):
        result.append(arr1[k + i] + arr2[i])
    if n > k + overlap_len:
        result.extend(arr1[k + overlap_len:])
    if m > overlap_len:
        result.extend(arr2[overlap_len:])
    
    return np.array([result])

def find_first_gt(arr, start, threshold=1):
    arr = arr.flatten()
    for i in range(start, len(arr)):
        if np.abs(arr[i]) > threshold:
            return i
    return -1

def ofdm_dec(ofdm, nfft=64, n_sc=29):
    ofdm_demod = np.fft.fft(ofdm, nfft)
    bits_est = np.zeros(n_sc)
    for j in range(1,n_sc+1):
        b = math.atan2(np.imag(ofdm_demod[j]), np.real(ofdm_demod[j]))
        if b >= math.pi / 2 or b <= -math.pi / 2:
            bits_est[j-1] = -1
        else:
            bits_est[j-1] = 1
    return bits_est.astype(int)

def sig_dec(all_sig, num_ofdm, ofdm_len):
    all_sym = []
    for user_idx in range(all_sig.shape[0]):
        user_sig = all_sig[user_idx,:]
        user_sig = user_sig.reshape(num_ofdm, ofdm_len)
        sym = []
        for ofdm_idx in range (user_sig.shape[0]):
            bits_est = ofdm_dec(user_sig[ofdm_idx,:])
            sym.extend(bits_est)
        all_sym.append(sym)

    return np.array(all_sym)

""" common parameters"""
ofdm_len = 512
n_sc = 29
sig_len = ofdm_len * args.num_ofdm
sym_len = n_sc * args.num_ofdm

""" Create model """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_mix = np.random.randn(args.num_ofdm, 1, ofdm_len)
test_sig = np.random.randn(args.num_ofdm, 1, ofdm_len)
tensor_x = torch.Tensor(test_mix)
tensor_y = torch.Tensor(test_sig)

test_dataset = TensorDataset(tensor_x,tensor_y)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

model_name = args.model
if model_name == 'waveunet':
    model = Waveunet(n_src=2, n_first_filter=20, depth=5)
elif model_name == 'OFDM_SepNet':
    model = OFDM_SepNet()
elif model_name == 'SepMamba':
    model = SepMambaCausal(dim=64)

model = model.to(device=device)

folder_name = f'{args.file_root}/checkpoints/{args.model}/{args.model}_SNRs_models/'
file_list = glob.glob(folder_name+"*")
file_list = sorted(file_list, key=lambda t: -os.stat(t).st_mtime)
file_list = [ fname for fname in file_list if fname.endswith('.ckpt')]
filename = file_list[0]
path_name = filename

loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
system = System(model, optimizer, loss, test_loader, test_loader)

ckpt = torch.load(path_name, map_location=device)
system.load_state_dict(ckpt['state_dict'], strict=False)

""" Signal Separation """
power = [1.0 + args.coeff_step*i for i in range(args.num_user)]
snr_tensor = torch.tensor([args.snr]).to(device=device)

mse_list = []
ber_list = []
flag_list = []

for _ in tqdm(range(args.num_sample)):
    np.random.shuffle(power)

    sigs, syms = gen_data_multi(user_num=args.num_user, ofdm_num=args.num_ofdm, power=power)
    sigs = np.array(sigs).real.reshape(-1, 1, sig_len)
    syms = np.array(syms).reshape(-1, 1, sym_len)

    cp = torch.randperm(ofdm_len-1)[:args.num_user-1] + 1

    sig_noisy = sigs[0,:]
    sig_noisy = sig_noisy.reshape(1, args.num_ofdm, ofdm_len)
    sig_noisy = add_awgn_noise(sig_noisy, args.snr)

    sigs_mix = sig_noisy.reshape(1,-1)
    for i in range(1, args.num_user):
        sigs_mix = add_with_offset(sigs_mix, sigs[i,:], cp[i-1])


    flag = True
    first_flag = True
    start = 0
    all_sig_clean = []
    all_start = [0]
    err_flag = 0
    count = 0

    while flag:
        if first_flag:
            end = start + ofdm_len
        else:
            num = count // user_num
            idx = count % user_num
            start = all_start[idx] + num * ofdm_len
            end = start + ofdm_len
        sigs_mix_target = sigs_mix[:,start:end]
        sigs_mix_target = np.array(sigs_mix_target).real.reshape(-1, 1, ofdm_len)
        tensor_sigs_mix = torch.Tensor(sigs_mix_target)
        system.eval()
        with torch.no_grad():
            tensor_sigs_mix = tensor_sigs_mix.to(device=device)
            sig_est = system(tensor_sigs_mix, snr_tensor)
        sig_est = sig_est.cpu().numpy()
        sig_clean = sig_est[0,0:1,:]
        sig_noise = sig_est[0,1:2,:]
        sig_clean = sig_clean.reshape(1, ofdm_len)
        sig_noise = sig_noise.reshape(1, ofdm_len)
        all_sig_clean.append(sig_clean)
        sigs_mix = sigs_mix.copy()
        sigs_mix[:,start:end] -= sig_noise[:]

        count += 1

        if first_flag:
            start = find_first_gt(sigs_mix, start)
            if start == -1:
                err_flag = -1
                flag = False
            elif start + sig_len == sigs_mix.shape[1]:
                    first_flag = False
                    all_start.append(start)
                    user_num = len(all_start)
            elif start + sig_len > sigs_mix.shape[1]:
                    err_flag = -2
                    flag = False
            else:
                all_start.append(start)
        if end == sigs_mix.shape[1]:
            flag = False
        

    if err_flag == 0:
        if user_num == args.num_user and len(all_sig_clean) == args.num_user*args.num_ofdm:
            all_sig_clean = np.array(all_sig_clean).squeeze()
            x_reshape = all_sig_clean.reshape(args.num_ofdm, args.num_user, ofdm_len).transpose(1, 0, 2)
            x_final = x_reshape.reshape(args.num_user, -1)

            diff = x_final - sigs.squeeze()
            squared_diff = diff ** 2
            mse = np.mean(squared_diff)

            sym_x = sig_dec(x_final, num_ofdm=args.num_ofdm, ofdm_len=ofdm_len)

            errors = (sym_x != syms.squeeze())
            num_err = np.count_nonzero(errors)
            total = sym_x.size
            ber = num_err / total
        else:
            err_flag = -3
            mse = -1000
            ber = -1000
    else:
        mse = -1000
        ber = -1000
    
    mse_list.append(mse)
    ber_list.append(ber)
    flag_list.append(err_flag)

df = pd.DataFrame({
    "MSE": mse_list,
    "BER": ber_list,
    "Flag": flag_list
})

filename = f"{args.file_root}/logs/results/{args.model}/NU{args.num_user}/{args.model}_NU{args.num_user}_NO{args.num_ofdm}_CS{args.coeff_step}_SNR{args.snr}.xlsx"
df.to_excel(filename, index=False)

print('Done!')



