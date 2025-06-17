import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime


data = pd.read_csv('house3_5devices_train.csv')
data['date'] = pd.to_datetime(data['time'], unit='s').dt.date
data['total_power'] = data[['lighting2', 'lighting5', 'lighting4', 'refrigerator', 'microwave']].sum(axis=1)
daily_data = data.groupby('date')['total_power'].mean().reset_index()
daily_data['day_of_month'] = pd.to_datetime(daily_data['date']).dt.day
daily_data['daily_energy_wh'] = daily_data['total_power'] * 24


features = []
targets = []

for i in range(1, len(daily_data)):
    # Input: [day of month, previous day's energy]
    feature = [
        daily_data['day_of_month'].iloc[i],
        daily_data['daily_energy_wh'].iloc[i-1]  # Previous day energy
    ]
    # Output: current energy
    target = daily_data['daily_energy_wh'].iloc[i]
    features.append(feature)
    targets.append(target)


features = np.array(features)
targets = np.array(targets)

scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)
targets_scaled = scaler_target.fit_transform(targets.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, targets_scaled, test_size=0.2, random_state=42
)
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)


class EnergyPredictor(nn.Module):
    def __init__(self):
        super(EnergyPredictor, self).__init__()

        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 8)  
        self.layer3 = nn.Linear(8, 1)  
        self.relu = nn.ReLU()  

    def forward(self, x):
    
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    

model = EnergyPredictor()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 750 
for epoch in range(epochs):
    model.train() 
    optimizer.zero_grad()  
    outputs = model(X_train).squeeze() 
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')


predictions = scaler_target.inverse_transform(predictions.numpy().reshape(-1, 1)).flatten()
y_test_orig = scaler_target.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

last_day = daily_data.iloc[-1]
last_energy = last_day['daily_energy_wh']
remaining_days = list(range(last_day['day_of_month'] + 1, 31))

future_predictions = []
current_energy = last_energy


for day in remaining_days:

    input_data = np.array([[day, current_energy]])
    input_scaled = scaler_features.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)
    
    model.eval()
    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()
    pred = scaler_target.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    
    future_predictions.append(pred)
    current_energy = pred


avg_monthly_energy = np.mean(future_predictions)
print(f'Predicted average daily energy for the rest of the month: {avg_monthly_energy:.2f} watt-hours')

plt.figure(figsize=(10, 5))
plt.plot(y_test_orig, label='Actual Energy (Test Data)')
plt.plot(predictions, label='Predicted Energy (Test Data)')
plt.title('Actual vs Predicted Daily Energy Usage')
plt.xlabel('Test Sample')
plt.ylabel('Energy (Watt-hours)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(remaining_days, future_predictions, label='Predicted Energy')
plt.title('Predicted Daily Energy for Rest of Month')
plt.xlabel('Day of Month')
plt.ylabel('Energy (Watt-hours)')
plt.legend()
plt.show()
