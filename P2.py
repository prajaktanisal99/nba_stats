import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
import warnings

# Suppress warnings
warnings.simplefilter("ignore")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)  # For GPU (if applicable)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess the data
COLS = [
    "Pos",
    "Age",
    "Tm",
    "G",
    "GS",
    "MP",
    "3P",
    "3PA",
    "FG",
    "FGA",
    "2P",
    "2PA",
    "FT",
    "FTA",
]
nba = pd.read_csv("nba_stats.csv")

positional_mapping = {
    position: index for index, position in enumerate(nba["Pos"].unique())
}
nba["Pos"] = nba["Pos"].map(positional_mapping)

target = nba["Pos"]
train = nba.drop(columns=COLS)

X_train, X_test, Y_train, Y_test = train_test_split(
    train, target, test_size=0.2, random_state=seed, stratify=target
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.long)


class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(MLPWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], output_dim)
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.tanh(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.tanh(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


input_dim = X_train.shape[1]
hidden_units = (64, 32)
output_dim = len(positional_mapping)
model = MLPWithDropout(input_dim, hidden_units, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.7)

epochs = 200
batch_size = 32

best_val_loss = float("inf")
patience = 5
counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)

    loss.backward()
    optimizer.step()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        Y_val_pred = model(X_test_tensor)
        val_loss = criterion(Y_val_pred, Y_test_tensor)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
        )

model.eval()
with torch.no_grad():
    Y_pred_train = model(X_train_tensor)
    _, predicted_train = torch.max(Y_pred_train, 1)

    Y_pred_test = model(X_test_tensor)
    _, predicted_test = torch.max(Y_pred_test, 1)

    train_accuracy = accuracy_score(Y_train_tensor, predicted_train)
    test_accuracy = accuracy_score(Y_test_tensor, predicted_test)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv_scores = []

for train_idx, val_idx in stratified_kfold.split(X_train, Y_train):
    model.train()
    optimizer.zero_grad()

    X_train_fold, X_val_fold = X_train_tensor[train_idx], X_train_tensor[val_idx]
    Y_train_fold, Y_val_fold = Y_train_tensor[train_idx], Y_train_tensor[val_idx]

    outputs = model(X_train_fold)
    loss = criterion(outputs, Y_train_fold)

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        Y_val_pred = model(X_val_fold)
        _, predicted_val = torch.max(Y_val_pred, 1)
        fold_accuracy = accuracy_score(Y_val_fold, predicted_val)
        cv_scores.append(fold_accuracy)

print(
    f"\n10-Fold Cross-validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
)

print("\n\n--------------------------------")
print("Dummy Dataset")
print("--------------------------------")

dummy_test = pd.read_csv("dummy_test.csv")
dummy_test["Pos"] = dummy_test["Pos"].map(positional_mapping)

Y_dummy_test = dummy_test["Pos"]
X_dummy_test = dummy_test.drop(columns=COLS)

X_dummy_test_scaled = scaler.transform(X_dummy_test)
X_dummy_test_tensor = torch.tensor(X_dummy_test_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    Y_dummy_pred = model(X_dummy_test_tensor)
    _, predicted_dummy = torch.max(Y_dummy_pred, 1)
    dummy_accuracy = accuracy_score(Y_dummy_test, predicted_dummy)
    print(f"Dummy Accuracy with Best Model: {dummy_accuracy:.4f}")


# OUTPUT
# Epoch 10/200, Loss: 1.4729, Val Loss: 1.4674
# Epoch 20/200, Loss: 1.3278, Val Loss: 1.3063
# Epoch 30/200, Loss: 1.2644, Val Loss: 1.1886
# Epoch 40/200, Loss: 1.1958, Val Loss: 1.1171
# Epoch 50/200, Loss: 1.1661, Val Loss: 1.0747
# Epoch 60/200, Loss: 1.1475, Val Loss: 1.0446
# Epoch 70/200, Loss: 1.0789, Val Loss: 1.0215
# Epoch 80/200, Loss: 1.0687, Val Loss: 1.0077
# Epoch 90/200, Loss: 1.0548, Val Loss: 0.9978
# Epoch 100/200, Loss: 1.0497, Val Loss: 0.9906
# Epoch 110/200, Loss: 1.0426, Val Loss: 0.9837
# Epoch 120/200, Loss: 1.0398, Val Loss: 0.9772
# Epoch 130/200, Loss: 1.0350, Val Loss: 0.9737
# Epoch 140/200, Loss: 1.0460, Val Loss: 0.9672
# Epoch 150/200, Loss: 1.0145, Val Loss: 0.9619
# Epoch 160/200, Loss: 0.9947, Val Loss: 0.9599
# Epoch 170/200, Loss: 1.0074, Val Loss: 0.9565
# Epoch 180/200, Loss: 1.0073, Val Loss: 0.9536
# Epoch 190/200, Loss: 0.9930, Val Loss: 0.9518
# Epoch 200/200, Loss: 0.9988, Val Loss: 0.9513
# Train Accuracy: 0.6199
# Test Accuracy: 0.6082

# 10-Fold Cross-validation Accuracy: 0.5817 ± 0.0658


# --------------------------------
# Dummy Dataset
# --------------------------------
# Dummy Accuracy with Best Model: 0.5631
