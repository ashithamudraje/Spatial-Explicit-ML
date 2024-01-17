import xarray as xr
import numpy as np
import torch.nn as nn
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import collections

# Open the dataset
training_data = xr.open_dataset("/netscratch/mudraje/spatial-explicit-ml/full_train.nc")

# Extract coordinates and input data
coordinates = training_data['coords'].values
input_data = training_data['input_data'].values
target = training_data['target'].values
# Check the shapes before concatenation
print("Coordinates shape:", coordinates.shape)
print("Input data shape:", input_data.shape)

# Reshape coordinates to match the dimensions of input data
coordinates_expanded = np.expand_dims(coordinates, axis=1)
coordinates_expanded = np.repeat(coordinates_expanded, input_data.shape[1], axis=1)

# Concatenate coordinates with input data along the last axis
coordinates_input = np.concatenate([input_data, coordinates_expanded], axis=-1)
# Check the shape after concatenation
print("Concatenated data shape:", coordinates_input.shape)

class TempCNN(nn.Module):
    def __init__(self, input_size, hidden_size, len_seq, num_classes, num_layers=2):
        super(TempCNN, self).__init__()
        self.conv1d_1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(hidden_size*2)

        self.lstm = nn.LSTM(hidden_size*2, hidden_size*4, num_layers, batch_first=True)
        self.conv1d_3 = nn.Conv1d(len_seq, len_seq*2, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.bn_2 = nn.BatchNorm1d(len_seq*2)
        self.relu4 = nn.ReLU()

        self.fc1 = nn.Linear((len_seq*2)*(hidden_size*4), 256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        # modified to accept batches and not only individual pixels

        N = x.shape[0]                  # batch size

        # === 1st Conv Layer ===
        out = self.conv1d_1(x)
        out = self.relu1(out)
        out = F.dropout(out, 0.4)

        # === 2nd Conv Layer ===
        out = self.conv1d_2(out)
        out = self.relu2(out)
        out = F.dropout(out, 0.4)
        out = self.bn_1(out)

        # === Swapping axis for LSTM input ===
        out = torch.swapaxes(out,1,2)

        # === LSTM ===
        out, _ = self.lstm(out)

        # === 3rd Conv Layer ===
        out = self.conv1d_3(out)
        out = self.relu3(out)
        out = F.dropout(out, 0.4)
        out = self.bn_2(out)

        # === Create input for FC Layer by reshaping ===
        out = out.reshape((N,-1))
        out = self.relu4(out)
        out = F.dropout(out, 0.4)

        # === 1st Fully Connected Layer ===
        out = self.fc1(out)
        out = self.relu5(out)
        out = F.dropout(out, 0.4)

        # === 2nd Fully Connected Layer ===
        out = self.fc2(out)

        return out

channels = coordinates_input.shape[2]
timesteps = coordinates_input.shape[1]
num_classes = 1  # Adjust based on your problem
hidden_size = 64
model = TempCNN(channels, hidden_size, timesteps, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_combined_tensor = torch.tensor(coordinates_input).float().to(device)
print(X_combined_tensor.shape)
Y_combined_tensor = torch.tensor(target).long().to(device)
print(Y_combined_tensor.shape)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_combined_tensor = X_combined_tensor.permute(0, 2, 1)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_combined_tensor, Y_combined_tensor, test_size=0.2, random_state=42)

# Create data loaders for training and validation
batch_size = 50
train_dataset = data_utils.TensorDataset(X_train, Y_train)
train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = data_utils.TensorDataset(X_val, Y_val)
val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    model.train()  # Set the model to training mode

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # labels = labels.unsqueeze(1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predictions = (outputs > 0).float()
        correct_predictions += torch.sum(predictions == labels.float())
        total_samples += labels.size(0)

    # Calculate training loss for the epoch
    train_loss = running_loss / len(train_loader)
    train_accuracy = (correct_predictions / total_samples).item()

    # Validation
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0
    val_predictions_list = []
    val_labels_list = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            # labels = labels.unsqueeze(1)
            val_loss = criterion(outputs, labels.float())
            val_running_loss += val_loss.item()

            val_predictions = (outputs > 0).float()
            val_correct_predictions += torch.sum(val_predictions == labels.float())
            val_total_samples += labels.size(0)

            # Collect predictions and true labels for confusion matrix
            val_predictions_list.append(val_predictions.cpu().numpy())
            val_labels_list.append(labels.float().cpu().numpy())

    # Calculate validation loss and accuracy for the epoch
    val_loss = val_running_loss / len(val_loader)
    val_accuracy = (val_correct_predictions / val_total_samples).item()

    # Flatten the lists of predictions and labels for the confusion matrix
    val_predictions_flat = np.concatenate(val_predictions_list).flatten()
    val_labels_flat = np.concatenate(val_labels_list).flatten()

    # Calculate and print the confusion matrix
    confusion = confusion_matrix(val_labels_flat, val_predictions_flat)
    plt.figure(figsize=(1, 1))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix (Epoch {epoch+1})")
    plt.show()

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    
def inference(testing_country):
  # Inference/Prediction
  model.eval()
  X_country_tensor = testing_country.permute(0, 2, 1)
  with torch.no_grad():
      Y_country_pred_probs = model(X_country_tensor)
      #Transforming the values into probabilities in the range [0, 1] by squashing the input values between these bounds.
      Y_country_pred_probs = torch.sigmoid(Y_country_pred_probs)
  return Y_country_pred_probs

# Load the testing dataset
testing_data = xr.open_dataset("/content/five_test.nc")

# Extract coordinates and input data
coordinates = testing_data['coords'].values
input_data = testing_data['input_data'].values

# Check the shapes before concatenation
print("Coordinates shape:", coordinates.shape)
print("Input data shape:", input_data.shape)

# Reshape coordinates to match the dimensions of input data
coordinates_expanded = np.expand_dims(coordinates, axis=1)
coordinates_expanded = np.repeat(coordinates_expanded, input_data.shape[1], axis=1)

# Concatenate coordinates with input data along the last axis
concatenated_data = np.concatenate([input_data, coordinates_expanded], axis=-1)

# Update the xarray dataset with the new concatenated data
testing_data['input_data'] = xr.DataArray(concatenated_data, dims=('identifier', 'input_data-D1', 'input_data-D2_new'))

# Check the shape after concatenation
print("Concatenated data shape:", concatenated_data.shape)
testing_data['input_data'].shape

# Get the unique countries
countries_list = testing_data['countries'].values

# Create a dictionary to store separate xarray Datasets for each country
country_datasets = {}

# Iterate over each country and create a separate Dataset
for country in countries_list:
    # Find the indices where the country variable matches the current country
    country_indices = (testing_data['countries'] == country).values

    # Use boolean indexing to select data for the current country
    country_datasets[country] = testing_data.isel(identifier=country_indices)

# Now, 'country_datasets' contains individual xarray Datasets for each country

Brazil_country = country_datasets['Brazil']['input_data'].values
print(Brazil_country.shape)
Brazil_country = torch.from_numpy(Brazil_country).float().to(device)
Brazil_pred_probs = inference(Brazil_country)
# Converting the tensor into numpy
Brazil_pred_probs= (Brazil_pred_probs).numpy()
print(Brazil_pred_probs)

Canada_country = country_datasets['Canada']['input_data'].values
print(Canada_country.shape)
Canada_country = torch.from_numpy(Canada_country).float().to(device)
Canada_pred_probs = inference(Canada_country)
# Converting the tensor into numpy
Canada_pred_probs= (Canada_pred_probs).numpy()
print(Canada_pred_probs)

France_country = country_datasets['France']['input_data'].values
print(France_country.shape)
France_country = torch.from_numpy(France_country).float().to(device)
France_pred_probs = inference(France_country)
# Converting the tensor into numpy
France_pred_probs= (France_pred_probs).numpy()
print(France_pred_probs)

India_country = country_datasets['India']['input_data'].values
print(India_country.shape)
India_country = torch.from_numpy(India_country).float().to(device)
India_pred_probs = inference(India_country)
# Converting the tensor into numpy
India_pred_probs= (India_pred_probs).numpy()
print(India_pred_probs)

Chile_country = country_datasets['Chile']['input_data'].values
print(Chile_country.shape)
Chile_country = torch.from_numpy(Chile_country).float().to(device)
Chile_pred_probs = inference(Chile_country)
# Converting the tensor into numpy
Chile_pred_probs= (Chile_pred_probs).numpy()
print(Chile_pred_probs)

def probabilities(Country_pred_probs):
  if len(Country_pred_probs.shape) > 1:  # if probabilities are given, extract the most likely class (0 or 1)
    Y_country_pred_final = np.argmax(Country_pred_probs)
  if np.any( (Country_pred_probs < 1) & (Country_pred_probs >0)): #transform probalities to 0 or 1
    Y_country_pred_final = 1*(Country_pred_probs >0.55).squeeze()
  return(Y_country_pred_final)

Y_Brazil_pred_final = probabilities(Brazil_pred_probs)
Y_Canada_pred_final = probabilities(Canada_pred_probs)
Y_France_pred_final = probabilities(France_pred_probs)
Y_India_pred_final = probabilities(India_pred_probs)
Y_Chile_pred_final = probabilities(Chile_pred_probs)

Brazil_target_country = country_datasets['Brazil']['target'].values
Canada_target_country = country_datasets['Canada']['target'].values
France_target_country = country_datasets['France']['target'].values
India_target_country = country_datasets['India']['target'].values
Chile_target_country = country_datasets['Chile']['target'].values

def perclass_accuracy(y_pred, y_true, class_names={}):
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)
    if len(class_names) == 0:
        classes = np.unique(y_true)
        class_names = {v: i for i, v in enumerate(classes)}

    acc_per_class = {}
    for name, value in class_names.items():
        mask_class = y_true == value
        acc_per_class[name] = np.mean(y_pred[mask_class] == y_true[mask_class])
    acc_per_class["average"] = np.mean(list(acc_per_class.values()))
    return acc_per_class

# Brazil
accuracy = perclass_accuracy(Y_Brazil_pred_final, Brazil_target_country)
print('Brazil:', accuracy)

# Canada
accuracy = perclass_accuracy(Y_Canada_pred_final, Canada_target_country)
print('Canada:', accuracy)

# France
accuracy = perclass_accuracy(Y_France_pred_final, France_target_country)
print('France:', accuracy)

# India
accuracy = perclass_accuracy(Y_India_pred_final, India_target_country)
print('India:', accuracy)

# Chile
accuracy = perclass_accuracy(Y_Chile_pred_final, Chile_target_country)
print('Chile:', accuracy)