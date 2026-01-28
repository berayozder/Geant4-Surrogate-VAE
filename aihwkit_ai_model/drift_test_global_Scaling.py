import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightNoiseType

# --- 1. DATA LOADING (Return to Linear Normalization) ---
print("🚀 Loading Data...")
try:
    df = pd.read_csv('./data/training_data.csv', header=None)
    data = df.iloc[:, 1:].values # Excluding EventID
    
    # Noise cleaning
    data[data < 0] = 0
    
    max_val = np.max(data)
    data_normalized = data / max_val
    
    # Automatically get data dimension
    input_dim = data.shape[1] 
    print(f"Data Ready. Input Dimension: {input_dim}, Max Value: {max_val:.2f} MeV")
    
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

# --- 2. VAE ARCHITECTURE ---
def create_analog_vae_network(input_size): # Input size added as parameter
    
    rpu_config = InferenceRPUConfig()
    rpu_config.forward.out_noise = 0.01 
    rpu_config.noise_model.read_noise = 0.05 

    class AnalogVAE(nn.Module):
        def __init__(self):
            super(AnalogVAE, self).__init__()
            
            # --- ENCODER ---
            # Using dynamic 'input_size' instead of 784
            self.fc1 = AnalogLinear(input_size, 16, rpu_config=rpu_config) # 400 too large, 16 sufficient
            self.fc21 = AnalogLinear(16, 3, rpu_config=rpu_config) # Latent size: 3
            self.fc22 = AnalogLinear(16, 3, rpu_config=rpu_config) 

            # --- DECODER ---
            self.fc3 = AnalogLinear(3, 16, rpu_config=rpu_config)
            self.fc4 = AnalogLinear(16, input_size, rpu_config=rpu_config)
           

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std

        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))

        def forward(self, x):
            # .view(-1, 784) ERROR FIXED. Data already comes flattened.
            mu, logvar = self.encode(x) 
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    return AnalogVAE()

# --- 3. TRAINING ---
model = create_analog_vae_network(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate increased back

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 0.5 * KLD

print("🧠 Training V3 Model (Linear Scale)...")
epochs = 50 # Sufficient duration
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
        optimizer.step()
        train_loss += loss.item()
        
    
    avg_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {avg_loss:.4f}')

# --- 4. RESULT VISUALIZATION (FINAL) ---
model.eval()
with torch.no_grad():
    val_data = test_dataset[:][0]
    
    noise = torch.randn(len(val_data), 3) 
    
    generated_data = model.decode(noise).numpy() * max_val
    real_data = val_data.numpy() * max_val

plt.figure(figsize=(12, 6))
plt.plot(np.mean(real_data, axis=0), label='Geant4 (Truth)', color='blue', linewidth=3, alpha=0.7)
plt.plot(np.mean(generated_data, axis=0), label='Analog AI (Simulated)', color='red', linestyle='--', linewidth=2)

plt.title('Analog Memristor Simulation: Geant4 vs AI')
plt.legend()
plt.grid(True)
plt.savefig('./results/analog_vae_result.png')
print("✅ Done! Check results.")

# --- 5. DRIFT (TIME EVOLUTION) ANALYSIS ---
print("\n⏳ Starting Time Drift Analysis...")

# Time schedule (in seconds)
times = [0, 1, 3600, 86400, 2592000] 
time_labels = ['t=0 (Now)', 't=1s', 't=1 Hour', 't=1 Day', 't=1 Month']
colors = ['red', 'orange', 'green', 'blue', 'purple']

plt.figure(figsize=(12, 6))

# Plot Reference Data
sample_data = test_dataset[:][0] 
real_data_plot = np.mean(sample_data.numpy() * max_val, axis=0)
plt.plot(real_data_plot, label='Geant4 (Ground Truth)', color='black', linewidth=3, linestyle='-')

# --- DRIFT COMPENSATION LOOP ---

# Reference baseline (energy level at t=0)
# We assume we read this from "reference resistors" at the chip edge.
baseline_energy = None 

for i, t in enumerate(times):
    print(f"   Testing drift at {time_labels[i]}...")
    
    # 1. Apply Drift
    if t > 0:
        for layer in model.modules():
            if hasattr(layer, 'drift_analog_weights'):
                layer.drift_analog_weights(t_inference=t)
    
    # 2. Make Prediction
    model.eval()
    with torch.no_grad():
        fixed_noise = torch.randn(len(sample_data), 3) 
        output = model.decode(fixed_noise).numpy() * max_val
        
        # --- COMPENSATION (NEW ADDITION) ---
        current_energy = np.mean(output)
        
        if t == 0:
            baseline_energy = current_energy # Save reference
            corrected_output = output
        else:
            # Simple Scaling: If energy increased too much, reduce; if decreased, increase.
            correction_factor = baseline_energy / current_energy
            corrected_output = output * correction_factor
            
            print(f"     -> Drift Correction Factor: {correction_factor:.3f}")

    # 3. Add to Plot (plotting corrected data)
    plt.plot(np.mean(corrected_output, axis=0), 
             label=f'Corrected AI ({time_labels[i]})', 
             color=colors[i], 
             linestyle='--', 
             alpha=0.8)

plt.title('Impact of Memristor Drift Over Time')
plt.xlabel('Calorimeter Layer ID')
plt.ylabel('Energy Deposition [MeV]')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('./results/drift_analysis_global_scaling.png')
print("✅ Drift analysis completed! Check 'drift_analysis_global_scaling.png'.")