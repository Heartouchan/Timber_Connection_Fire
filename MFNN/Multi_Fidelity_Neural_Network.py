import numpy as np
import torch
import random
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Set random seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(0)

# Define models
class Unit(nn.Module):
    def __init__(self, in_N, out_N):
        super(Unit, self).__init__()
        self.L = nn.Linear(in_N, out_N)
    def forward(self, x):
        return torch.nn.functional.leaky_relu(self.L(x))

class NN1(nn.Module):  # Nonlinear NN
    def __init__(self, in_N, width, depth, out_N):
        super(NN1, self).__init__()
        self.stack = nn.ModuleList([Unit(in_N, width)] + [Unit(width, width) for _ in range(depth)] + [nn.Linear(width, out_N)])
    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x

class NN2(nn.Module):  # Linear NN
    def __init__(self, in_N, width, depth, out_N):
        super(NN2, self).__init__()
        self.stack = nn.ModuleList([nn.Linear(in_N, width)] + [nn.Linear(width, width) for _ in range(depth)] + [nn.Linear(width, out_N)])
    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x

# Utility functions
def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def train_model(model, optimizer, x_train, y_train, x_val, y_val, epochs=2000, name="Model"):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        inputs, targets = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.tensor(x_val, dtype=torch.float32))
                val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.float32).view(-1, 1))
                print(f'{name}: Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

def evaluate_model(model, x, y, label="Model"):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(x, dtype=torch.float32)
        outputs = model(inputs).numpy()
    r2 = r2_score(y, outputs)
    print(f'R^2 score for {label}: {r2:.6f}')
    return outputs

# Load and preprocess data
LF_data = pd.read_csv('LF_data_load_ratio_10.csv', encoding='gbk', header=None)
HF_data = pd.read_csv('HF_data_load_ratio_10.csv', encoding='gbk', header=None)
x_lo, y_lo = LF_data.iloc[:, :-1].values, LF_data.iloc[:, -1].values
x_hi, y_hi = HF_data.iloc[:, :-1].values, HF_data.iloc[:, -1].values

scaler = MinMaxScaler()
x_scaled_lo = scaler.fit_transform(x_lo)
x_scaled_hi = scaler.transform(x_hi)

# Step 1: Train LFNN
x_train_lo, x_val_lo, y_train_lo, y_val_lo = train_test_split(x_scaled_lo, y_lo, test_size=0.2, random_state=0)
model_l = NN1(9, 90, 3, 1)
model_l.apply(weights_init)
optimizer_l = optim.Adam(model_l.parameters(), lr=0.01, amsgrad=True)
train_model(model_l, optimizer_l, x_train_lo, y_train_lo, x_val_lo, y_val_lo, name="LFNN")

output_l = evaluate_model(model_l, x_scaled_lo, y_lo, label="LFNN")

# Plot LFNN results
plt.figure(figsize=(10,8))
plt.scatter(y_lo, output_l, label="LFNN", color='black')
plt.plot([y_lo.min(), y_lo.max()], [y_lo.min(), y_lo.max()], 'r-')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.grid()
plt.legend()
plt.show()

# Step 2: Projection NN
pred_hi = model_l(torch.tensor(x_scaled_hi, dtype=torch.float32)).detach()
pred_lo = model_l(torch.tensor(x_scaled_lo, dtype=torch.float32)).detach()
x_mh = torch.cat((torch.tensor(x_scaled_hi, dtype=torch.float32), pred_hi), dim=1)
x_ml = torch.cat((torch.tensor(x_scaled_lo, dtype=torch.float32), pred_lo), dim=1)

alpha = torch.tensor([0.35], requires_grad=True)
model3, model4 = NN1(10, 9, 2, 1), NN2(10, 9, 2, 1)
model3.apply(weights_init)
model4.apply(weights_init)
optimizer_proj = optim.Adam([{'params': model3.parameters()}, {'params': model4.parameters()}, {'params': alpha}], lr=0.02, amsgrad=True)

# Train projection models
criterion = nn.MSELoss()
for epoch in range(1000):
    inputs, targets = torch.tensor(x_mh, dtype=torch.float32), torch.tensor(y_hi, dtype=torch.float32).view(-1,1)
    optimizer_proj.zero_grad()
    outputs = alpha * model3(inputs) + (1 - alpha) * model4(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer_proj.step()
    if epoch % 100 == 0:
        print(f'Projection: Epoch {epoch}, Loss: {loss.item():.6f}')

# Step 3: Enhanced Data Generation
y_lo_enhanced = (alpha * model3(x_ml) + (1 - alpha) * model4(x_ml)).detach().view(-1)
x_hl = np.vstack((x_hi, x_lo))
y_hl = np.hstack((y_hi, y_lo_enhanced.numpy()))
x_hl_scaled = scaler.fit_transform(x_hl)

# Step 4: Final Training MFNN
model_lhn = NN1(9, 90, 3, 1)
model_lhl = NN2(9, 90, 2, 1)
model_lhn.apply(weights_init)
model_lhl.apply(weights_init)
optimizer_final = optim.Adam([{'params': model_lhn.parameters()}, {'params': model_lhl.parameters()}, {'params': alpha}], lr=0.01, amsgrad=True)

x_train_hl, x_val_hl, y_train_hl, y_val_hl = train_test_split(torch.tensor(x_hl_scaled, dtype=torch.float32), torch.tensor(y_hl, dtype=torch.float32), test_size=0.2, random_state=0)

for epoch in range(2000):
    optimizer_final.zero_grad()
    outputs = alpha * model_lhn(x_train_hl) + (1 - alpha) * model_lhl(x_train_hl)
    loss = criterion(outputs, y_train_hl.view(-1,1))
    loss.backward()
    optimizer_final.step()
    if epoch % 100 == 0:
        print(f'MFNN: Epoch {epoch}, Loss: {loss.item():.6f}')

# Step 5: Evaluate MFNN
with torch.no_grad():
    pred_hi_final = alpha * model_lhn(torch.tensor(x_scaled_hi, dtype=torch.float32)) + (1 - alpha) * model_lhl(torch.tensor(x_scaled_hi, dtype=torch.float32))
    r2_final = r2_score(y_hi, pred_hi_final.numpy())
    print(f'MFNN R^2: {r2_final:.6f}')
    plt.figure(figsize=(10,8))
    plt.scatter(y_hi, pred_hi_final.numpy(), label="MFNN", color='black')
    plt.plot([y_hi.min(), y_hi.max()], [y_hi.min(), y_hi.max()], 'r-')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.grid()
    plt.legend()
    plt.show()

# Exclusive Validation
HF_data_val = pd.read_csv('HF_data_validation_load_ratio_10.csv', encoding='gbk', header=None)
x_hi_val, y_hi_val = HF_data_val.iloc[:, :-1].values, HF_data_val.iloc[:, -1].values
x_scaled_hi_val = scaler.transform(x_hi_val)
pred_val = alpha * model_lhn(torch.tensor(x_scaled_hi_val, dtype=torch.float32)) + (1 - alpha) * model_lhl(torch.tensor(x_scaled_hi_val, dtype=torch.float32))
r2_val = r2_score(y_hi_val, pred_val.detach().numpy())
print(f'Validation R^2: {r2_val:.6f}')

plt.figure(figsize=(10,8))
plt.scatter(y_hi_val, pred_val.detach().numpy(), label="Validation", color='black')
plt.plot([y_hi_val.min(), y_hi_val.max()], [y_hi_val.min(), y_hi_val.max()], 'r-')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.grid()
plt.legend()
plt.show()
