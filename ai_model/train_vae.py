import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

# --- 1. DATA LOADING (Return to Linear Normalization) ---
print("🚀 Loading Data...")
try:
    df = pd.read_csv('./data/training_data.csv', header=None)
    data = df.iloc[:, 1:].values # Excluding EventID
    
    # Noise cleaning
    data[data < 0] = 0
    
    # Linear Normalization (Like V1 - Safest)
    max_val = np.max(data)
    data_normalized = data / max_val
    
    print(f"Data Ready. Max Value: {max_val:.2f} MeV")
    
except FileNotFoundError:
    print("ERROR: 'training_data.csv' not found!")
    exit()

tensor_x = torch.Tensor(data_normalized)
dataset = TensorDataset(tensor_x)

# Train/Test Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- 2. VAE ARCHITECTURE (V3 - Hybrid) ---
# V1 simplicity + V2 power
class VAE_V3(nn.Module):
    def __init__(self, input_dim=20, latent_dim=6): # Latent 5->6 (Increased slightly)
        super(VAE_V3, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(), # Classic ReLU instead of LeakyReLU (More stable)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Output between 0-1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# --- 3. TRAINING ---
model = VAE_V3(input_dim=20, latent_dim=6)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate increased back

def loss_function(recon_x, x, mu, logvar):
    # MSE Loss (Mean Squared Error) - Ensures pixel (energy) exact match
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD weight 0.5 (Found balance)
    return BCE + 0.5 * KLD

print("🧠 Training V3 Model (Linear Scale)...")
epochs = 80 # Sufficient duration
train_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        data_batch = batch[0]
        optimizer.zero_grad()
        recon, mu, logvar = model(data_batch)
        loss = loss_function(recon, data_batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {avg_loss:.4f}')

# --- 4. RESULT VISUALIZATION (FINAL) ---
model.eval()
with torch.no_grad():
    val_data = test_dataset[:][0]
    
    # AI Prediction
    noise = torch.randn(len(val_data), 6)
    generated_data = model.decoder(noise).numpy() * max_val
    real_data = val_data.numpy() * max_val

plt.figure(figsize=(12, 6))
# Real (Blue)
plt.plot(np.mean(real_data, axis=0), label='Geant4 (Truth)', color='blue', linewidth=3, alpha=0.7)
plt.fill_between(range(20), 
                 np.mean(real_data, axis=0) - np.std(real_data, axis=0),
                 np.mean(real_data, axis=0) + np.std(real_data, axis=0), color='blue', alpha=0.1)

# AI (Red)
plt.plot(np.mean(generated_data, axis=0), label='DeepCalo AI (Generated)', color='red', linestyle='--', linewidth=2)
plt.fill_between(range(20), 
                 np.mean(generated_data, axis=0) - np.std(generated_data, axis=0),
                 np.mean(generated_data, axis=0) + np.std(generated_data, axis=0), color='red', alpha=0.1)

plt.title('Final Validation: Geant4 vs AI Surrogate Model')
plt.xlabel('Calorimeter Layer ID')
plt.ylabel('Energy Deposition [MeV]')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('./results/geant4_vs_ai_final.png')

# Save model
torch.save(model.state_dict(), 'ai_model/vae_model.pth')
print("✅ V3 Model trained. Check 'geant4_vs_ai_final.png'!")