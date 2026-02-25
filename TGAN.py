import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils import spectral_norm
from glob import glob
import random
import os
import torch.nn.functional as F

# ==========================================
# 1. Reproducibility & Device Setup
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
FILE_PATH_PATTERN = "/content/Data/402510/*.csv"

def load_and_process_data(pattern, time_steps=288):
    files = glob(pattern)
    dataset = []

    for file in files:
        df = pd.read_csv(file)
        dataset.append(df)

    full_data = pd.concat(dataset, axis=0, ignore_index=True)

    # Normalize
    data_values = full_data[['Flow (Veh/5 Minutes)']].values.astype(float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data_values)

    # Sequence Creation
    X = []
    for i in range(0, len(data_scaled) - time_steps + 1, 288):
        X.append(data_scaled[i:i + time_steps])

    return np.array(X), scaler

print("Processing data...")
X_train, scaler = load_and_process_data(FILE_PATH_PATTERN)

tensor_data = torch.tensor(X_train, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

print(f"Data loaded. Shape: {X_train.shape}")

# ==========================================
# 3. Latent Space Components
# ==========================================
class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        return torch.tanh(self.out(x))


class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        return torch.tanh(self.out(x))


# ==========================================
# 4. GAN Components
# ==========================================
class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len, hidden_dim, embedding_dim):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim, seq_len * hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)

        self.tconv1 = TemporalConvBlock(hidden_dim, hidden_dim, dilation=2)
        self.tconv2 = TemporalConvBlock(hidden_dim, hidden_dim, dilation=4)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z).view(B, self.seq_len, self.hidden_dim)

        positions = torch.arange(self.seq_len, device=z.device)
        x = x + self.pos_emb(positions)

        x = self.tconv1(x)
        x = self.tconv2(x)

        x = self.out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, 128,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True)

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv1d(256, 128, 3, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv1d(128, 64, 3, padding=1)),
            nn.LeakyReLU(0.2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.mean(x, dim=2)
        return self.fc(x)


# ==========================================
# 5. Training Function
# ==========================================
def train_latent_gan(embedder, recovery, generator, discriminator, loader, args):

    ae_opt = optim.Adam(
        list(embedder.parameters()) + list(recovery.parameters()),
        lr=1e-3
    )

    g_opt = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []

    mse_loss = nn.MSELoss()

    print(">>> Phase 1: Training Autoencoder...")

    for epoch in range(args['ae_epochs']):
        epoch_loss = 0
        for (real_seq,) in loader:
            real_seq = real_seq.to(device)

            h = embedder(real_seq)
            rec = recovery(h)

            loss = mse_loss(rec, real_seq)

            ae_opt.zero_grad()
            loss.backward()
            ae_opt.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"AE Epoch {epoch+1}: Loss {epoch_loss/len(loader):.6f}")

    print("\n>>> Phase 2: Training GAN...")

    for epoch in range(args['gan_epochs']):
        epoch_g = 0
        epoch_d = 0
        batches = 0

        for (real_seq,) in loader:
            real_seq = real_seq.to(device)
            batch_size = real_seq.size(0)

            # --- Train Discriminator ---
            h_real = embedder(real_seq).detach()

            z = torch.randn(batch_size, args['latent_dim'], device=device)
            h_fake = generator(z)

            d_real = discriminator(h_real)
            d_fake = discriminator(h_fake.detach())

            d_loss = (
                torch.mean(torch.relu(1 - d_real)) +
                torch.mean(torch.relu(1 + d_fake))
            )

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # --- Train Generator ---
            z = torch.randn(batch_size, args['latent_dim'], device=device)
            h_fake = generator(z)
            d_fake_g = discriminator(h_fake)

            g_loss = -torch.mean(d_fake_g)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            epoch_g += g_loss.item()
            epoch_d += d_loss.item()
            batches += 1

        g_losses.append(epoch_g / batches)
        d_losses.append(epoch_d / batches)

        if (epoch + 1) % 100 == 0:
            print(f"GAN Epoch {epoch+1} | D: {epoch_d/batches:.4f} | G: {epoch_g/batches:.4f}")

    return g_losses, d_losses


# ==========================================
# 6. Hyperparameters
# ==========================================
PARAMS = {
    'seq_len': 288,
    'input_dim': 1,
    'embedding_dim': 64,
    'hidden_dim': 64,
    'latent_dim': 250,
    'ae_epochs': 100,
    'gan_epochs': 2000
}

embedder = Embedder(PARAMS['input_dim'], PARAMS['embedding_dim']).to(device)
recovery = Recovery(PARAMS['embedding_dim'], PARAMS['input_dim']).to(device)
generator = Generator(PARAMS['latent_dim'], PARAMS['seq_len'],
                      PARAMS['hidden_dim'], PARAMS['embedding_dim']).to(device)
discriminator = Discriminator(PARAMS['embedding_dim']).to(device)

g_losses, d_losses = train_latent_gan(
    embedder, recovery, generator, discriminator, data_loader, PARAMS
)

# ==========================================
# 7. Generation
# ==========================================
print("\nGenerating synthetic samples...")
generator.eval()
recovery.eval()

set_seed(42)

with torch.no_grad():
    z_sample = torch.randn(30, PARAMS['latent_dim'], device=device)
    h_synthetic = generator(z_sample)
    x_synthetic = recovery(h_synthetic)

    synth_np = x_synthetic.cpu().numpy().reshape(-1, 1)
    real_scale_synth = scaler.inverse_transform(synth_np).reshape(30, 288, 1)

print(f"Generated shape: {real_scale_synth.shape}")
print("Sample generation complete.")
