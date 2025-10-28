# UNetMamba1D_v1中的hidden_dim由原来的32变为16
import torch.nn.functional as F
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class SMamba(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.LN = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        x_in = x
        x = self.mamba(x)
        x = self.LN(x)
        x = self.linear(x)
        return x + x_in
    
class BMamba(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba1 = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba2 = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.LN1 = nn.LayerNorm(dim)
        self.LN2 = nn.LayerNorm(dim)
        self.linear = nn.Linear(2*dim, dim)

    def forward(self, x):
        # x is (B, L, D)
        x1_mamba_out = self.mamba1(x)
        x1 = self.LN1(x1_mamba_out)
        x1 = x + x1

        x_flipped = torch.flip(x, dims=[1])
        x2_mamba_out = self.mamba2(x_flipped)
        x2 = self.LN2(x2_mamba_out)
        x2 = x_flipped + x2

        x = torch.concatenate([x1, x2], dim=2)
        x = self.linear(x)
        return x

class MambaBlock(nn.Module): # MFMB
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm_smamba = nn.LayerNorm(dim//2)
        self.norm_bmamba = nn.LayerNorm(dim//2)
        
        self.BMamba = BMamba(dim//2, d_state, d_conv, expand)
        self.SMamba = SMamba(dim//2, d_state, d_conv, expand)
        self.dconv1 = nn.Conv1d(dim, dim//2, kernel_size=5, padding=2 * 1, dilation=1)
        self.dconv2 = nn.Conv1d(dim//2, dim, kernel_size=5, padding=2 * 2, dilation=2)
        self.dconv3 = nn.Conv1d(dim, dim//2, kernel_size=5, padding=2 * 4, dilation=4)
        self.dconv4 = nn.Conv1d(dim//2, dim, kernel_size=5, padding=2 * 8, dilation=8)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x is (B, D, L)
        # LTMU
        x_res1 = x
        x = self.dconv1(x) # (B, D/2, L)
        x = x.permute(0, 2, 1) # (B, L, D/2)
        x = self.norm_smamba(x)
        x = self.SMamba(x) # (B, L, D/2)
        x = x.permute(0, 2, 1) # (B, D/2, L)
        x = self.dconv2(x) # (B, D, L)
        x = x + x_res1

        # GCMU
        x_res2 = x
        x = self.dconv3(x) # (B, D/2, L)
        x = x.permute(0, 2, 1) # (B, L, D/2)
        x = self.norm_bmamba(x)

        x = self.BMamba(x) # (B, L, D/2)
        x = x.permute(0, 2, 1) # (B, D/2, L)
        x = self.dconv4(x) # (B, D, L)
        x = x + x_res2

        x = self.dropout(x)
        return x

class SNRRecalibration(nn.Module): # SAFS
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_dim = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels + 1, hidden_dim)  # +1 for SNR
        self.fc2 = nn.Linear(hidden_dim, channels)

    def forward(self, x, snr):
        # x: [B, C, L], snr: [B]
        B, C, L = x.shape
        context = x.mean(dim=-1)  # [B, C]
        snr = snr.view(B, 1)      # [B, 1]
        context = torch.cat([context, snr], dim=1)  # [B, C+1]

        scale = self.fc1(context)
        scale = F.relu(scale)
        scale = torch.sigmoid(self.fc2(scale))  # [B, C]
        scale = scale.unsqueeze(-1)             # [B, C, 1]

        return x * scale


class OFDM_SepNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1,
                 hidden_dim=16,
                 num_layers=4,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
        )

        # Encoder
        self.encoder_mamba_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(num_layers):
            self.encoder_mamba_blocks.append(MambaBlock(current_dim, mamba_d_state, mamba_d_conv, mamba_expand))
            if i < num_layers - 1:
                self.downsamples.append(nn.Conv1d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
                current_dim *= 2

        # Bottleneck
        self.bottleneck_mamba = MambaBlock(current_dim, mamba_d_state, mamba_d_conv, mamba_expand)

        # Decoder
        self.decoder_mamba_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(num_layers):
            skip_dim = hidden_dim * (2 ** (self.num_layers - 1 - i))
            self.upsamples.append(nn.ConvTranspose1d(current_dim, skip_dim, kernel_size=4, stride=2, padding=1))
            combined_dim = skip_dim * 2
            self.decoder_mamba_blocks.append(MambaBlock(combined_dim, mamba_d_state, mamba_d_conv, mamba_expand))
            current_dim = combined_dim 

        self.snr_recalib = SNRRecalibration(current_dim)

        self.final_reducing_conv = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.final_conv_clean = nn.Conv1d(hidden_dim, output_channels, kernel_size=7, padding=3)
        self.final_conv_noisy = nn.Conv1d(hidden_dim, output_channels, kernel_size=7, padding=3)

    def forward(self, x, snr):
        skips = []
        x = self.initial_conv(x)

        # Encoder
        for i in range(self.num_layers):
            x = self.encoder_mamba_blocks[i](x)
            skips.append(x)
            if i < self.num_layers - 1:
                x = self.downsamples[i](x)

        # Bottleneck
        x = self.bottleneck_mamba(x)

        # Decoder
        for i in range(self.num_layers):
            x = self.upsamples[i](x)
            skip_connection = skips[self.num_layers - 1 - i]
            if x.shape[-1] != skip_connection.shape[-1]:
                x = F.interpolate(x, size=skip_connection.shape[-1], mode='linear', align_corners=True)
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder_mamba_blocks[i](x)

        # Recalibration with SNR
        x_recalibrated = self.snr_recalib(x, snr)

        # Clean signal path
        clean_x = self.final_reducing_conv(x_recalibrated)
        clean_out = self.final_conv_clean(clean_x)

        # Noisy signal path
        noisy_x = self.final_reducing_conv(x)
        noisy_out = self.final_conv_noisy(noisy_x)

        final_output = torch.cat([clean_out, noisy_out], dim=1) # (B, 2, L)

        return final_output

if __name__ == "__main__":
    model = OFDM_SepNet(hidden_dim=16).cuda()
    
    inp = torch.randn(2, 1, 512).cuda()
    snr_input = torch.tensor([10.0, 5.0]).cuda()

    out = model(inp, snr_input)
    print("Output shape:", out.shape) 

    predicted_clean_signal = out[:, 0, :]
    predicted_noisy_original_signal = out[:, 1, :]
    print("Predicted clean signal shape:", predicted_clean_signal.shape)
    print("Predicted noisy original signal shape:", predicted_noisy_original_signal.shape)

