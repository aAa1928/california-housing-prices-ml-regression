# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%
class Model(nn.Module):
    def __init__(self, in_features: int = 9, h1: int = 64, h2: int = 32, out_features: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

def graph(epochs: int, losses: list[float], save: bool = False) -> None:
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save:
        plt.savefig('loss_plot.png')
    plt.show()

torch.manual_seed(392)

# %%
dataframe = pd.read_csv(r"california-housing-prices-dataset\housing.csv")

print(dataframe.head())
print("Missing values per column:\n", dataframe.isnull().sum())

# Removes 207 of 20,640 rows with missing values in the dataset, specifically in the total_bedrooms column.
dataframe = dataframe.dropna()
print(f"Rows after dropping missing values: {len(dataframe)}")
# %%

dataframe['ocean_proximity'] = dataframe['ocean_proximity'].map({value: index for index, value in enumerate(dataframe['ocean_proximity'].unique())})
X = dataframe.drop("median_house_value", axis=1).values.astype('float32')

# Apply StandardScaler
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# %%
y = dataframe["median_house_value"].values

# Scale the target variable
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=392)
print(len(X_train), len(X_test))

# %%
# Convert data to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)


# %%
# Initialize model and training parameters
model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.15)

epochs = 1000
losses = []
print('Training...')
for i in range(1, epochs + 1):
    y_pred = model(X_train)
    
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    
    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss.item():.2f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Training finished!\n')

graph(epochs, losses, save=True)

print('Testing...')
with torch.no_grad():
    y_pred = model(X_test)
    
    # Convert predictions and test data back to original scale
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    test_loss = criterion(torch.FloatTensor(y_pred_orig), torch.FloatTensor(y_test_orig))
    
    for i in range(len(y_test)):
        actual = y_test_orig[i].item()
        predicted = y_pred_orig[i].item()
        percentage_diff = abs((predicted - actual) / actual * 100)
        print(f'Test {i+1:3d} | Predicted: ${predicted:10,.2f} | Actual: ${actual:10,.2f} | Diff: {percentage_diff:6.1f}%')
    
    print(f'\nTest MSE: {test_loss.item():,.2f}')
    print(f'Test RMSE: ${torch.sqrt(test_loss).item():,.2f}')
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    percentage_errors = [(abs(y_pred_orig[i].item() - y_test_orig[i].item()) / y_test_orig[i].item() * 100) for i in range(len(y_test))]
    mape = sum(percentage_errors) / len(percentage_errors)
    print(f'Test MAPE: {mape:.1f}%')

torch.save(model.state_dict(), 'house_price_regression_model.pth')