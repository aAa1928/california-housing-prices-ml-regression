# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
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

def graph(epochs: int, losses: list[float]) -> None:
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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
X
# %%
y = dataframe["median_house_value"].values
y

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
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
losses = []
print('Training...')
for i in range(1, epochs + 1):
    y_pred = model(X_train)
    
    loss = criterion(y_pred, y_train)
    print(f'test: {loss}')
    losses.append(loss.item())
    
    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss.item():.2f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Training finished!\n')

graph(epochs, losses)

print('Testing...')
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    
    for i in range(len(y_test)):
        print(f'Test {i+1} | Predicted: ${y_pred[i].item():,.2f} | Actual: ${y_test[i].item():,.2f}')
    
    print(f'\nTest MSE: {test_loss.item():,.2f}')
    print(f'Test RMSE: ${torch.sqrt(test_loss).item():,.2f}')

# Save the model
torch.save(model.state_dict(), 'house_price_regression_model.pth')