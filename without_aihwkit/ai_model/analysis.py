import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CURRENT MODEL ARCHITECTURE ---
class VAE_V3(nn.Module):
    def __init__(self, input_dim=20, latent_dim=6): # Latent dim became 6
        super(VAE_V3, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
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
            nn.Sigmoid() 
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

print("📂 Loading Data and Model...")

# Load Real Data
try:
    # Adjust data path according to your structure (data/training_data.csv)
    df = pd.read_csv('./data/training_data.csv', header=None) 
    real_data = df.iloc[:, 1:].values
    # Noise cleaning (Negatives to zero)
    real_data[real_data < 0] = 0
    max_val = np.max(real_data)
except FileNotFoundError:
    print("ERROR: './data/training_data.csv' not found. Check the path!")
    exit()

# Load Trained Model (Using V3 Class)
model = VAE_V3(input_dim=20, latent_dim=6) # Latent dim 6!
try:
    # Adjust model path according to your structure (ai_model/vae_model.pth)
    model_path = 'ai_model/vae_model.pth' 
    # If in the same folder, just write 'vae_model.pth'
    # model_path = 'vae_model.pth' 
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"✅ Model ({model_path}) loaded successfully!")
except FileNotFoundError:
    print("ERROR: 'vae_model.pth' not found.")
    exit()
except RuntimeError as e:
    print(f"ERROR: Model architecture mismatch! Is the class here the same as in train_vae.py?\nDetail: {e}")
    exit()

# --- 2. ARTIFICIAL DATA GENERATION (INFERENCE) ---
print("🧠 AI is Generating Data...")
num_samples = 10000
with torch.no_grad():
    # Latent Space (6 dimensional)
    noise = torch.randn(num_samples, 6)
    generated_data = model.decoder(noise).numpy() * max_val

# --- 3. DETAILED ANALYSIS PLOTS ---
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3)

# A) Total Energy
ax1 = fig.add_subplot(gs[0, 0])
real_sum = np.sum(real_data, axis=1)
gen_sum = np.sum(generated_data, axis=1)
sns.histplot(real_sum, color="blue", label="Geant4", kde=True, ax=ax1, stat="density", alpha=0.4)
sns.histplot(gen_sum, color="red", label="AI (V3)", kde=True, ax=ax1, stat="density", alpha=0.4)
ax1.set_title("Total Energy Distribution", fontweight='bold')
ax1.legend()

# B) Correlation Difference
ax2 = fig.add_subplot(gs[0, 1])
corr_real = np.corrcoef(real_data.T)
corr_gen = np.corrcoef(generated_data.T)
sns.heatmap(corr_real - corr_gen, cmap="coolwarm", vmin=-0.2, vmax=0.2, ax=ax2)
ax2.set_title("Correlation Difference (Real - AI)", fontweight='bold')

# C) Single Events
ax3 = fig.add_subplot(gs[0, 2])
for i in range(3):
    idx = np.random.randint(0, num_samples)
    ax3.plot(generated_data[idx], linestyle='--', marker='o', alpha=0.7, label=f'AI Event {i+1}')
ax3.plot(np.mean(real_data, axis=0), color='black', linewidth=2, label='Mean Real Profile')
ax3.set_title("Single AI Generated Events", fontweight='bold')
ax3.legend(fontsize='small')

# D) Box Plot
ax4 = fig.add_subplot(gs[1, :])
df_real = pd.DataFrame(real_data)
df_real['Source'] = 'Geant4'
df_gen = pd.DataFrame(generated_data)
df_gen['Source'] = 'AI'
# Take only first 1000 events for faster plotting
df_combined = pd.concat([df_real.iloc[:1000], df_gen.iloc[:1000]])
df_melted = df_combined.melt(id_vars=['Source'], var_name='Layer', value_name='Energy')

sns.boxplot(x='Layer', y='Energy', hue='Source', data=df_melted, ax=ax4, 
            palette={'Geant4': 'blue', 'AI': 'red'}, showfliers=False)
ax4.set_title("Layer-wise Energy Distribution (Box Plot)", fontweight='bold')

plt.tight_layout()
# Save to results folder (create if not exists)
import os
if not os.path.exists('results'):
    os.makedirs('results')
plt.savefig("./results/detailed_analysis.png")
print("✅ Detailed analysis saved to 'results/detailed_analysis.png'.")
# plt.show() # You can close this if running on a server