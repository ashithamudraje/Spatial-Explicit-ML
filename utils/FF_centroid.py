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
import sys
import lightning
import rasterio
import torchgeo
import os
import copy



def load_data(data_path):
    data = xr.open_dataset(data_path)
    return data


class CentroidNN(nn.Module):    # Define the feedforward neural network for centroid processing
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

class DataProcessor:
    
    def process_centroid_data(training_data, centroid_df):
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
            # Forward pass through the dynamically created centroid processing model for the current country
            centroid_tensors_per_country.append(centroid_values_tensor)
        # Combine the processed values for all countries
        centroid_tensor_combined = torch.cat(centroid_tensors_per_country, dim=0)
        # Print or use the combined centroid tensors as needed
        print("Centroid Tensor shape:", centroid_tensor_combined.shape)

        return sorted_dict_keys, centroid_tensor_combined
    
    def process_input_data(training_data, centroid_dict):
        input_tensors_per_country = []
        for country, centroid_values in centroid_dict.items():
            # Find identifiers for the current country
            country_identifiers = training_data['identifier'].where(training_data['countries'] == country, drop=True).values
            # Update 'input_data' variable with the new values
            input_dataset = training_data.sel(identifier=country_identifiers)
            existing_values = input_dataset['input_data'].values
            # Convert to tensors
            existing_values_tensor = torch.tensor(existing_values)
            input_tensors_per_country.append(existing_values_tensor)
        input_tensor_combined = torch.cat(input_tensors_per_country, dim=0)
        print("Input Tensor shape:", input_tensor_combined.shape)

        return input_tensor_combined
    
    def normalize_centroid_data(centroid_tensor):
        # Calculate mean and std for centroid tensor
        centroid_mean = centroid_tensor.mean(dim=0)
        centroid_std = centroid_tensor.std(dim=0)

        centroid_std[centroid_std == 0] = 1e-6

        # Normalize centroid tensor
        normalized_centroid_tensor = (centroid_tensor - centroid_mean) / centroid_std
        print("Normalized Centroid Tensor shape:", normalized_centroid_tensor.shape)

        return normalized_centroid_tensor
    
    def normalize_input_data(input_tensor):
        # Calculate mean and std for input tensor
        input_mean = input_tensor.mean(dim=(0, 1))
        input_std = input_tensor.std(dim=(0, 1))

        # Handle zero standard deviation
        input_std[input_std == 0] = 1e-6

        # Normalize input tensor
        normalized_input_tensor = (input_tensor - input_mean) / input_std
        print("Normalized Input Tensor shape:", normalized_input_tensor.shape)
        
        return normalized_input_tensor
    
    def get_target_lables(training_data):
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
            print("Shape of target labels:", input_target_per_country.shape)
        except Exception as e:
            print(f"Error: {e}")
        return input_target_per_country
    
    def testing_process_data(testing_data, centroid_df):
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

        centroid_dict = {}
        # Iterate over the country DataFrames and fetch centroid information
        for country, country_df in country_datasets.items():
            # Extract the country's centroid information
            country_centroid = centroid_df[country == centroid_df['COUNTRY'] ][['longitude', 'latitude']].values[0]
        # Assign the centroid values to each specific country
            centroid_dict[country] = country_centroid
        # Display the centroid dictionary
        print("\nCentroid Dictionary:")
        for country, centroid_info in centroid_dict.items():
            print(f"{country}: {centroid_info}")
        # Loop through each country in the dataset
        centroid_test_tensors_per_country = {}
        for country, centroid_values in centroid_dict.items():
            # Find identifiers for the current country
            country_identifiers = testing_data['identifier'].where(testing_data['countries'] == country, drop=True).values
            # Broadcast centroid values to the correct shape
            broadcasted_centroid_values = np.tile(np.expand_dims(centroid_values, axis=0), (len(country_identifiers), 1))
            # Convert centroid values to PyTorch tensor
            centroid_values_tensor = torch.tensor(broadcasted_centroid_values, dtype=torch.float32)
            centroid_test_tensors_per_country[country] = centroid_values_tensor
        print(centroid_test_tensors_per_country['France'].shape)

        return country_datasets, centroid_test_tensors_per_country

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
    
class ConcatenatedNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, sensor, centroid, num_layers=2, sequence_length=20):
        super(ConcatenatedNN, self).__init__()
        self.conv1d_1 = nn.Conv1d(1, hidden_size, kernel_size=3, stride=1, padding=1).double()
        self.relu1 = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, stride=1, padding=1).double()
        self.relu2 = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(hidden_size*2)

        self.lstm = nn.LSTM(input_size, hidden_size*4, num_layers, batch_first=True)
        self.conv1d_3 = nn.Conv1d(hidden_size*4, hidden_size*8, kernel_size=3, stride=1, padding=1).double()
        self.relu3 = nn.ReLU()
        self.bn_2 = nn.BatchNorm1d(hidden_size*8)

        # Calculate the input size for the fully connected layers dynamically
        lstm_output_size = hidden_size*8
        self.fc1 = nn.Linear(lstm_output_size * sequence_length, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

        self.centroid = centroid
        self.sensor = sensor

    def forward(self, x_sensor, x_centroid):

        output_centroid = self.centroid(x_centroid)
        output_sensor = self.sensor(x_sensor.permute(0, 2, 1))
        x = torch.cat([output_sensor, output_centroid], dim=1)

        N = x.shape[0]

        # 1st Conv Layer
        out = self.conv1d_1(x.double().unsqueeze(1).contiguous())  # Cast input to Double
        out = self.relu1(out)
        out = F.dropout(out, 0.4)

        # 2nd Conv Layer
        out = self.conv1d_2(out)
        out = self.relu2(out)
        out = F.dropout(out, 0.4)
        # Cast the input tensor to the same data type as the batch normalization layer
        out = self.bn_1(out.to(self.bn_1.running_mean.dtype))

       # LSTM
        out, _ = self.lstm(out)
        out = out.permute(0, 2, 1).contiguous()

        # 3rd Conv Layer
        out = self.conv1d_3(out.double())
        out = self.relu3(out)
        out = F.dropout(out, 0.4)
        out = self.bn_2(out.to(self.bn_2.running_mean.dtype))

        # Create input for FC Layer by reshaping
        out = out.reshape((N, -1))
        out = self.relu4(out)
        out = F.dropout(out, 0.4)


        # Adjust the input size of the first fully connected layer based on the actual shape of out
        self.fc1 = nn.Linear(out.size(1), 256)

        # 1st Fully Connected Layer
        out = self.fc1(out)
        out = self.relu4(out)
        out = F.dropout(out, 0.4)

        # 2nd Fully Connected Layer
        out = self.fc2(out)
        out = F.softmax(out, dim=1)

        return out

def train_and_evaluate_model(concatenated_model, model, centroid_model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, save_path ):
    best_val_accuracy = 0.0
    counter = 0

    # Training loop with early stopping
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0
        concatenated_model.train()  # Set the model to training mode
        model.train()
        centroid_model.train()

        for (input_sensor, input_centroid, labels) in train_loader:
            optimizer.zero_grad()

            outputs = concatenated_model(input_sensor, input_centroid)
            labels = labels.view(-1)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Get the class indices with maximum probability

            #predictions = (outputs > 0.45).float()
            train_correct_predictions += torch.sum(predicted == labels.float())
            train_total_samples += labels.size(0)

        # Calculate training loss and accuracy for the epoch
        train_loss = running_loss / len(train_loader)
        train_accuracy = (train_correct_predictions / train_total_samples).item()

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
            for (input_sensor, input_centroid, labels) in val_loader:

                outputs = concatenated_model(input_sensor,input_centroid)
                labels = labels.view(-1)

                val_loss = criterion(outputs, labels.long())
                val_running_loss += val_loss.item()

                _, val_predicted = torch.max(outputs.data, 1)
                val_correct_predictions += torch.sum(val_predicted == labels)
                val_total_samples += labels.size(0)

                # Collect predictions and true labels for the confusion matrix
                val_predictions_list.append(val_predicted.cpu().numpy())
                val_labels_list.append(labels.float().cpu().numpy())

        # Calculate validation loss and accuracy for the epoch
        val_loss = val_running_loss / len(val_loader)
        val_accuracy = (val_correct_predictions / val_total_samples).item()

        # Print validation accuracy
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Check for improvement in validation accuracy
        if val_accuracy > best_val_accuracy:
            # Save the model with the best validation accuracy
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(concatenated_model)  # Create a deep copy of the model
            torch.save(best_model, save_path)
            counter = 0  # Reset the patience counter
        else:
            counter += 1  # Increment the patience counter
            if counter >= patience:     # Check for early stopping
                print(f"Early stopping after {epoch+1} epochs of no improvement.")
                break

    # Load the best model
    best_model = torch.load(save_path)
    print("Model loaded with the best validation accuracy.")

    return best_model 
 

def inference(test_data_sensor, test_data_centroid, concatenated_model):
    """
    Perform inference for testing data using the saved model.
    """
    concatenated_model.eval()
    predictions_final_list =[]
    # Forward pass
    with torch.no_grad():
        outputs = concatenated_model(test_data_sensor, test_data_centroid)
        _, predictions_list = torch.max(outputs.data, 1)
        predictions_final_list.append(predictions_list.cpu().numpy())
    # Concatenate and flatten predictions
    return np.concatenate(predictions_final_list).flatten()

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

# Main program
if __name__ == "__main__":
    # Load data
    training_data = load_data("/netscratch/mudraje/spatial-explicit-ml/dataset/full_train.nc")
    testing_data = load_data("/netscratch/mudraje/spatial-explicit-ml/dataset/five_test.nc")
    centroid_df = pd.read_csv("/netscratch/mudraje/spatial-explicit-ml/countries.csv")

    # Preprocess centroid data
    centroid_dict, centroid_tensor_combined = DataProcessor.process_centroid_data(training_data, centroid_df)

    # Preprocess input data
    input_tensor_combined = DataProcessor.process_input_data(training_data, centroid_dict)

    # Normalize data
    normalized_centroid_tensor = DataProcessor.normalize_centroid_data(centroid_tensor_combined)
    normalized_input_tensor = DataProcessor.normalize_input_data(input_tensor_combined)

    #Target labels
    input_target_per_country = DataProcessor.get_target_lables(training_data)
    
    # Define model parameters
    features = normalized_input_tensor.shape[2]
    timesteps = normalized_input_tensor.shape[1]
    num_classes = 64
    hidden_size = 20
    sensor_model = CNN(features, hidden_size, timesteps, num_classes)

    # Assuming centroids for each country have 2 features (longitude, latitude)
    input_size = 2
    hidden_size = 20  # You can adjust this based on your problem
    output_size = 64  # Adjust based on your problem
    centroid_model = CentroidNN(input_size, hidden_size, output_size)

    # Concatenate models
    input_size_concatenated = 128
    num_classes_concatenated = 2
    hidden_size_concatenated = 10
    concatenated_model = ConcatenatedNN(input_size_concatenated, hidden_size_concatenated, num_classes_concatenated, sensor_model, centroid_model )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_combined_tensor = torch.tensor(normalized_input_tensor).float().to(device)
    centroid_tensor_combined_values = torch.tensor(normalized_centroid_tensor).float().to(device)
    Y_combined_tensor = torch.tensor(input_target_per_country).long().to(device)

    # Define loss function and optimizer
    criterion_concatenated = nn.CrossEntropyLoss()
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
    val_loader_combined = data_utils.DataLoader(val_dataset_combined, batch_size=batch_size, shuffle=False)

    # Preprocess testing data
    country_datasets, centroid_dict = DataProcessor.testing_process_data(testing_data, centroid_df)
    test_normalized_input_data = {}
    test_normalized_centroid_data = {}

    for country, _ in country_datasets.items():
        print(f'>>>>>>>>>>>>>>>>>>>>>>>{country}<<<<<<<<<<<<<<<<<<<<<<<<<')
        input_data = country_datasets[country]['input_data'].values
        input_data = torch.tensor(input_data, dtype=torch.float32)
        test_normalized_input_data[country] = DataProcessor.normalize_input_data(input_data)
        centroid_values = DataProcessor. normalize_centroid_data(centroid_dict[country])
        test_normalized_centroid_data[country] = centroid_values
        
    for i in range(5):
        # Train and evaluate the model
        print(f'-------------------------------------------------EXPERIMENT {i+1}:-------------------------------')
        num_epochs = 20
        patience = 6
        save_path = f'/netscratch/mudraje/spatial-explicit-ml/scripts/models/centroid/FFcentroid_best_model_exp_{i+1}.pt'  # Define save path for this experiment
        model = train_and_evaluate_model(concatenated_model, sensor_model, centroid_model, train_loader_combined, val_loader_combined, criterion_concatenated,
                             optimizer_concatenated, num_epochs, patience, save_path)
    
        print('-------------------------------------TRAINING COMPLETED--------------------------------------------')

  
        # Perform inference for each country
        for country, _ in country_datasets.items():
            predictions = inference(test_normalized_input_data[country], test_normalized_centroid_data[country], model)
            target_country = country_datasets[country]['target'].values
            accuracy = perclass_accuracy(predictions, target_country)
            print(f'{country}: {accuracy}')

        print('---------------------------------------TESTING COMPLETED-----------------------------------')
    testing_data.close()
    training_data.close()