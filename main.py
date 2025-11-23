# vae_ae_fashion_mnist.py
# Requirements: torch, torchvision, numpy, scikit-learn, matplotlib
# Run: python vae_ae_fashion_mnist.py
import os
import math
import numpy as np
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from typing import Tuple

# -------------------------
# Config / Hyperparameters
# -------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Data and training hyperparams
batch_size = 128
lr = 1e-3
num_epochs = 30           # adjust as compute allows (20-50 typical)
latent_dim = 16           # try 8,16,32 in hyperparam search
hidden_dim = 256
image_size = 28 * 28
beta = 1.0                # beta-VAE coefficient (1.0 = standard VAE). Try 0.5,1.0,2.0

# The 'normal' class: label 0 = T-shirt/top in Fashion-MNIST
normal_class = 0

# -------------------------
# Data preparation
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(), # [0,1]
])

# Download datasets
trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Filter training to normal class only
train_idx = [i for i, (_, y) in enumerate(trainset) if y == normal_class]
train_normal = Subset(trainset, train_idx)

# For testing we keep full testset (both normal and anomalies)
X_test = torch.stack([img for img, _ in testset])   # shape (N,1,28,28)
y_test = torch.tensor([label for _, label in testset])

# Convert test images to flattened vectors for convenience
X_test_flat = X_test.view(X_test.size(0), -1)

train_loader = DataLoader(train_normal, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# -------------------------
# Model definitions
# -------------------------
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=16):
        super().__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc_out = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        # For numerical stability, we will clamp logvar in reparam step

    def encode(self, x):
        h = self.relu(self.enc_fc1(x))
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Numerical stability: clamp logvar to a reasonable range
        logvar_clamped = torch.clamp(logvar, min=-20.0, max=5.0)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, logvar_clamped

    def decode(self, z):
        h = self.relu(self.dec_fc1(z))
        x_recon = torch.sigmoid(self.dec_fc_out(h))  # outputs in [0,1]
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, logvar = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=16):
        super().__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc_out = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.enc_fc1(x))
        z = self.relu(self.enc_fc2(h))
        h2 = self.relu(self.dec_fc1(z))
        x_recon = torch.sigmoid(self.dec_fc_out(h2))
        return x_recon

# -------------------------
# Loss functions
# -------------------------
mse_loss = nn.MSELoss(reduction='sum')  # sum over pixels for ELBO

def elbo_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    ELBO = - E_q[log p(x|z)] + KL(q(z|x) || p(z))
    We use MSE for reconstruction term (sum over dimensions).
    KL term for diagonal Gaussians has closed form.
    Return total loss (to minimize), recon_loss, kl_loss
    """
    # Reconstruction loss (sum)
    recon_loss = mse_loss(recon_x, x)

    # KL divergence between q(z|x) ~ N(mu, sigma^2) and p(z) ~ N(0, I)
    # KL = 0.5 * sum( mu^2 + sigma^2 - 1 - log(sigma^2) )
    # note: logvar is log(sigma^2)
    kl_element = mu.pow(2) + torch.exp(logvar) - 1.0 - logvar
    kl_loss = 0.5 * torch.sum(kl_element)

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss

# For AE we use MSE (sum or mean). Use sum to be comparable to VAE recon sum.
def ae_loss(recon_x, x):
    return mse_loss(recon_x, x)

# -------------------------
# Training and utilities
# -------------------------
def train_vae(model: VAE, optimizer, train_loader, epochs=num_epochs, beta=1.0):
    model.train()
    history = {'loss': [], 'recon': [], 'kl': []}
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            loss, recon_l, kl_l = elbo_loss(recon, imgs, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()
        history['loss'].append(total_loss / len(train_loader.dataset))
        history['recon'].append(total_recon / len(train_loader.dataset))
        history['kl'].append(total_kl / len(train_loader.dataset))
        print(f"VAE Epoch {epoch+1}/{epochs} | Loss: {history['loss'][-1]:.4f} Recon: {history['recon'][-1]:.4f} KL: {history['kl'][-1]:.4f}")
    return history

def train_ae(model: Autoencoder, optimizer, train_loader, epochs=num_epochs):
    model.train()
    history = {'loss': []}
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = ae_loss(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history['loss'].append(total_loss / len(train_loader.dataset))
        print(f"AE Epoch {epoch+1}/{epochs} | Loss: {history['loss'][-1]:.4f}")
    return history

@torch.no_grad()
def compute_reconstruction_errors_vae(model: VAE, X: torch.Tensor) -> np.ndarray:
    model.eval()
    X = X.to(device)
    recon, mu, logvar = model(X)
    # Use MSE per example (mean or sum). We'll compute per-sample MSE (mean across pixels).
    errors = torch.mean((recon - X) ** 2, dim=1).cpu().numpy()
    return errors

@torch.no_grad()
def compute_reconstruction_errors_ae(model: Autoencoder, X: torch.Tensor) -> np.ndarray:
    model.eval()
    X = X.to(device)
    recon = model(X)
    errors = torch.mean((recon - X) ** 2, dim=1).cpu().numpy()
    return errors

def prepare_test_tensors():
    # Flatten test images and create tensors for the whole test set
    Xt = X_test_flat  # defined earlier
    return Xt

# -------------------------
# Train models
# -------------------------
def run_training_and_evaluation():
    # Initialize models
    vae = VAE(input_dim=image_size, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    ae = Autoencoder(input_dim=image_size, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

    opt_vae = optim.Adam(vae.parameters(), lr=lr)
    opt_ae = optim.Adam(ae.parameters(), lr=lr)

    print("Training VAE...")
    _ = train_vae(vae, opt_vae, train_loader, epochs=num_epochs, beta=beta)

    print("\nTraining AE baseline...")
    _ = train_ae(ae, opt_ae, train_loader, epochs=num_epochs)

    # Prepare test tensors
    Xt = prepare_test_tensors().to(device)

    # Compute reconstruction errors (per-sample MSE)
    print("\nComputing reconstruction errors (VAE)...")
    vae_errors = compute_reconstruction_errors_vae(vae, Xt)
    print("Computing reconstruction errors (AE)...")
    ae_errors = compute_reconstruction_errors_ae(ae, Xt)

    # Labels: normal = 0, anomalies = 1
    y_true = (y_test != normal_class).numpy().astype(int)

    # Create an anomaly score: higher means more anomalous -> use reconstruction error
    y_scores_vae = vae_errors.copy()
    y_scores_ae = ae_errors.copy()

    # Compute ROC-AUC
    auc_vae = roc_auc_score(y_true, y_scores_vae)
    auc_ae = roc_auc_score(y_true, y_scores_ae)
    print(f"VAE AUC-ROC: {auc_vae:.4f}")
    print(f"AE  AUC-ROC: {auc_ae:.4f}")

    # Determine threshold from normal test samples: mean + k*std (k=3 default)
    normal_mask = (y_test == normal_class).numpy()
    normal_errors_vae = y_scores_vae[normal_mask]
    normal_errors_ae = y_scores_ae[normal_mask]

    k = 3.0
    thresh_vae = normal_errors_vae.mean() + k * normal_errors_vae.std()
    thresh_ae = normal_errors_ae.mean() + k * normal_errors_ae.std()
    print(f"Threshold (VAE) = mean + {k}*std = {thresh_vae:.6f}")
    print(f"Threshold (AE)  = mean + {k}*std = {thresh_ae:.6f}")

    # Evaluate at threshold: compute TPR, FPR, accuracy
    def threshold_metrics(y_true, scores, thr):
        y_pred = (scores > thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        acc = (tp + tn) / len(y_true)
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'tpr': tpr, 'fpr': fpr, 'acc': acc}

    metrics_vae = threshold_metrics(y_true, y_scores_vae, thresh_vae)
    metrics_ae  = threshold_metrics(y_true, y_scores_ae, thresh_ae)
    print("Metrics at threshold (VAE):", metrics_vae)
    print("Metrics at threshold (AE):", metrics_ae)

    # Save ROC curves to files
    def plot_roc(y_true, y_scores, title, fname):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0,1], [0,1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(fname)
        plt.close()

    plot_roc(y_true, y_scores_vae, 'VAE ROC', 'vae_roc.png')
    plot_roc(y_true, y_scores_ae,  'AE ROC',  'ae_roc.png')

    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save(vae.state_dict(), "models/vae.pth")
    torch.save(ae.state_dict(), "models/ae.pth")

    # Return results dictionary for printing/reporting
    results = {
        'auc_vae': auc_vae,
        'auc_ae': auc_ae,
        'thresh_vae': thresh_vae,
        'thresh_ae': thresh_ae,
        'metrics_vae': metrics_vae,
        'metrics_ae': metrics_ae,
        'vae_errors': vae_errors,
        'ae_errors': ae_errors,
        'y_true': y_true
    }
    return results

if __name__ == "__main__":
    results = run_training_and_evaluation()
    # Print summary
    print("\n--- Summary ---")
    print(f"VAE AUC-ROC: {results['auc_vae']:.4f}")
    print(f"AE  AUC-ROC: {results['auc_ae']:.4f}")
    print(f"Threshold (VAE): {results['thresh_vae']:.6f}")
    print(f"Threshold (AE):  {results['thresh_ae']:.6f}")
    print("ROC plots saved: vae_roc.png, ae_roc.png")
    print("Models saved to ./models/")
