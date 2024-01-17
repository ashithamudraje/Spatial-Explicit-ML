
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
print(training_data)
# Load the centroid data
centroid_df = pd.read_csv("/netscratch/mudraje/spatial-explicit-ml/countries.csv")
# print(training_data['input_data'][0].values)
print(training_data['identifier'][0].values)
training_data['countries'][0].values
training_data['coords'][0].values

# Create a dictionary to store centroid values for each country
centroids_dict = {}

# Loop through each country in the dataset
for country in training_data['countries'].values:
    # Extract the country's centroid information
    country_centroid = centroid_df[country == centroid_df['COUNTRY'] ][['longitude', 'latitude']].values[0]

    # Assign the centroid values to each specific country
    centroids_dict[country] = country_centroid
sorted_dict_keys = dict(sorted(centroids_dict.items()))

centroid_tensors_per_country = []
for country, centroid_values in sorted_dict_keys.items():
    # Find identifiers for the current country
    country_identifiers = training_data['identifier'].where(training_data['countries'] == country, drop=True).values
    # Broadcast centroid values to the correct shape
    broadcasted_centroid_values = np.tile(np.expand_dims(centroid_values, axis=0), (len(country_identifiers), 1))

    # Convert centroid values to PyTorch tensor
    centroid_values_tensor = torch.tensor(broadcasted_centroid_values, dtype=torch.float32)

    centroid_tensors_per_country.append(centroid_values_tensor)

# Combine the processed values for all countries
centroid_tensor_combined = torch.cat(centroid_tensors_per_country, dim=0)

# Print or use the combined centroid tensors as needed
print("Combined Centroid Tensor Shape:", centroid_tensor_combined.shape)

# Define the feedforward neural network for centroid processing
class CentroidNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CentroidNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# Assuming centroids for each country have 2 features (longitude, latitude)
input_size = 2
hidden_size = 20  # You can adjust this based on your problem
output_size = 64  # Adjust based on your problem
centroid_model = CentroidNN(input_size, hidden_size, output_size)

input_tensors_per_country = []

for country, centroid_values in sorted_dict_keys.items():
    # Find identifiers for the current country
    country_identifiers = training_data['identifier'].where(training_data['countries'] == country, drop=True).values

    # Update 'input_data' variable with the new values
    input_dataset = training_data.sel(identifier=country_identifiers)
    existing_values = input_dataset['input_data'].values

    # Convert to tensors
    existing_values_tensor = torch.tensor(existing_values)
    input_tensors_per_country.append(existing_values_tensor)
input_tensor_combined = torch.cat(input_tensors_per_country, dim=0)
print("Combined Input Tensor Shape:", input_tensor_combined.shape)

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, len_seq, num_classes, num_layers=2):
        super(CNN, self).__init__()
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
        # Adjust the dimensions based on the actual shape of out
        out = out.permute(0, 2, 1)  # Swap axes to match the expected shape
        out = out.reshape((N, -1))
        out = self.relu4(out)
        out = F.dropout(out, 0.4)

        # === 1st Fully Connected Layer ===
        # Adjust the input size based on the actual shape of out
        out = self.fc1(out)
        out = self.relu5(out)
        out = F.dropout(out, 0.4)

        # === 2nd Fully Connected Layer ===
        out = self.fc2(out)

        return out

features = input_tensor_combined.shape[2]
timesteps = input_tensor_combined.shape[1]
num_classes = 64 # Adjust based on your problem
hidden_size = 20
model = CNN(features, hidden_size, timesteps, num_classes)

# Get the list of unique countries from the dataset
countries_list = np.unique(training_data['countries'].values)
# Initialize an empty list to store target values for each country
target_list = []

# Loop through each country and extract the target values
for country in countries_list:
    # Get the indices of rows corresponding to the current country
    country_indices = np.where(training_data['countries'].values == country)[0]

    # Extract target values for the current country
    target_values = training_data['target'].values[country_indices]

    # Convert to a PyTorch tensor and append to the list
    target_list.append(torch.Tensor(target_values))

# Concatenate the list of tensors into a single tensor
try:
    input_target_per_country = torch.cat(target_list)
    # Print or use the input_target_per_country tensor as needed
    print("Target Tensor Shape:", input_target_per_country.shape)
except Exception as e:
    print(f"Error: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_combined_tensor = torch.tensor(input_tensor_combined).float().to(device)

centroid_tensor_combined_values = torch.tensor(centroid_tensor_combined).float().to(device)

Y_combined_tensor = torch.tensor(input_target_per_country).long().to(device)

print("X_combined_tensor Shape:", X_combined_tensor.shape)
print("Centroid_tensor_combined_values Shape:", centroid_tensor_combined_values.shape)
print("Y_combined_tensor Shape:", Y_combined_tensor.shape)

class ConcatenatedNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, sequence_length=20):  # Add sequence_length
        super(ConcatenatedNN, self).__init__()
        self.conv1d_1 = nn.Conv1d(1, hidden_size, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(hidden_size*2)

        self.lstm = nn.LSTM(input_size, hidden_size*4, num_layers, batch_first=True)  # Fix the input size here
        self.conv1d_3 = nn.Conv1d(hidden_size*4, hidden_size*8, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.bn_2 = nn.BatchNorm1d(hidden_size*8)

        # Calculate the input size for the fully connected layers dynamically
        lstm_output_size = hidden_size*8  # Output size of the LSTM layer
        self.fc1 = nn.Linear(lstm_output_size * sequence_length, 256)  # Correct the input size
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        N = x.shape[0]

        # 1st Conv Layer
        out = self.conv1d_1(x.unsqueeze(1).contiguous())  # Add channel dimension
        out = self.relu1(out)
        out = F.dropout(out, 0.4)

        # 2nd Conv Layer
        out = self.conv1d_2(out)
        out = self.relu2(out)
        out = F.dropout(out, 0.4)
        out = self.bn_1(out)

        # LSTM
        out, _ = self.lstm(out)
        out = out.permute(0, 2, 1).contiguous()
        # 3rd Conv Layer
        out = self.conv1d_3(out)
        out = self.relu3(out)
        out = F.dropout(out, 0.4)
        out = self.bn_2(out)

        # Create input for FC Layer by reshaping
        out = out.reshape((N, -1))
        out = self.relu4(out)
        out = F.dropout(out, 0.4)

        # 1st Fully Connected Layer
        out = self.fc1(out)
        out = self.relu4(out)
        out = F.dropout(out, 0.4)

        # 2nd Fully Connected Layer
        out = self.fc2(out)

        return out

# Assuming concatenated_tensor is your input tensor
input_size_concatenated = 128
num_classes_concatenated = 1
hidden_size_concatenated = 10
concatenated_model = ConcatenatedNN(input_size_concatenated, hidden_size_concatenated, num_classes_concatenated)



# Define loss function and optimizer
criterion_concatenated = nn.BCEWithLogitsLoss()
optimizer_concatenated = torch.optim.Adam(concatenated_model.parameters(), lr=0.001)

# Combine the two datasets into a zip object
combined_datasets = list(zip(X_combined_tensor, centroid_tensor_combined_values))

# Split the data into training and validation sets
train_data, val_data, Y_train, Y_val = train_test_split(combined_datasets, Y_combined_tensor, test_size=0.2, random_state=42)

# Unzip the datasets
X_train_sensor, X_train_centroid = zip(*train_data)
X_val_sensor, X_val_centroid = zip(*val_data)

# Combine the sensor and centroid data into a tuple for each set
train_dataset_combined = data_utils.TensorDataset(torch.stack(X_train_sensor), torch.stack(X_train_centroid), Y_train)
val_dataset_combined = data_utils.TensorDataset(torch.stack(X_val_sensor), torch.stack(X_val_centroid), Y_val)

# Create data loaders for training and validation
batch_size = 50
train_loader_combined = data_utils.DataLoader(train_dataset_combined, batch_size=batch_size, shuffle=True)
val_loader_combined = data_utils.DataLoader(val_dataset_combined, batch_size=batch_size, shuffle=True)



# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    concatenated_model.train()  # Set the model to training mode
    model.train()
    centroid_model.train()

    for (input_sensor, input_centroid, labels) in train_loader_combined:
        optimizer_concatenated.zero_grad()

        output_centroid = centroid_model(input_centroid)
        output_sensor = model(input_sensor.permute(0, 2, 1))

        output_concatenated = torch.cat([output_sensor, output_centroid], dim=1)
        outputs = concatenated_model(output_concatenated)

        loss = criterion_concatenated(outputs, labels.float())
        loss.backward()
        optimizer_concatenated.step()
        running_loss += loss.item()

        predictions = (outputs > 0).float()
        correct_predictions += torch.sum(predictions == labels.float())
        total_samples += labels.size(0)

    # Calculate training loss for the epoch
    train_loss = running_loss / len(train_loader_combined)
    train_accuracy = (correct_predictions / total_samples).item()

    # Validation
    concatenated_model.eval()  # Set the model to evaluation mode
    model.eval()
    centroid_model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0
    val_predictions_list = []
    val_labels_list = []

    with torch.no_grad():
        for (input_sensor, input_centroid, labels) in val_loader_combined:
            output_centroid = centroid_model(input_centroid)
            output_sensor = model(input_sensor.permute(0, 2, 1))

            output_concatenated = torch.cat([output_sensor, output_centroid], dim=1)
            outputs = concatenated_model(output_concatenated)

            val_loss = criterion_concatenated(outputs, labels.float())
            val_running_loss += val_loss.item()

            val_predictions = (outputs > 0).float()
            val_correct_predictions += torch.sum(val_predictions == labels.float())
            val_total_samples += labels.size(0)

            # Collect predictions and true labels for the confusion matrix
            val_predictions_list.append(val_predictions.cpu().numpy())
            val_labels_list.append(labels.float().cpu().numpy())

    # Calculate validation loss and accuracy for the epoch
    val_loss = val_running_loss / len(val_loader_combined)
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