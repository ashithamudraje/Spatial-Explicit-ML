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
import copy
      

def load_data(data_path):
    data = xr.open_dataset(data_path)
    return data

    

class DataProcessor:
    
    def process_coordinates_data(training_data, input_data):
        coordinates = training_data['coords'].values

        print("Coordinates shape before concatenating with input data:", coordinates.shape)
        
        # Reshape coordinates to match the dimensions of input data
        coordinates_expanded = np.expand_dims(coordinates, axis=1)
        coordinates_expanded = np.repeat(coordinates_expanded, input_data.shape[1], axis=1)

        # Concatenate coordinates with input data along the last axis
        coordinates_input = np.concatenate([input_data, coordinates_expanded], axis=-1)
        # Check the shape after concatenation
        print("Concatenated coordinate and input data shape:", coordinates_input.shape)
        coordinates_input_tensor = torch.tensor(coordinates_input, dtype=torch.float32)

        return coordinates_input_tensor
    
    def process_input_data(training_data):
        input_values = training_data['input_data'].values
        input_tensor_combined = torch.tensor(input_values, dtype=torch.float32)
        print("Input Tensor shape:", input_tensor_combined.shape)

        return input_tensor_combined
    
    def normalize_input_coordinate_data(coordinates_input_tensor):
        # Calculate mean and std for coordinates tensor
        coordinates_input_mean = coordinates_input_tensor.mean(dim=(0, 1))
        coordinates_input_std = coordinates_input_tensor.std(dim=(0, 1))

        coordinates_input_std[coordinates_input_std == 0] = 1e-6
        # Normalize coordinates tensor
        normalized_coordinates_input_tensor = (coordinates_input_tensor - coordinates_input_mean) / coordinates_input_std

        # Print or use the normalized tensors as needed
        print("Normalized Coordinates and Input Tensor shape:")
        print(normalized_coordinates_input_tensor.shape)

        
        return normalized_coordinates_input_tensor
    
    def get_target_lables(training_data):
        target_values = training_data['target'].values
        input_target_per_country = torch.tensor(target_values, dtype=torch.float32)
        print("Shape of target labels:", input_target_per_country.shape)

        return input_target_per_country
    
    def testing_process_data(testing_data):
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
            
        return country_datasets

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
        out = F.softmax(out, dim=1)

        return out

def train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, save_path):

    # Initialize variables for early stopping
    best_val_accuracy = 0.0
    counter = 0


    # Training loop with early stopping
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0
        model.train()       # Set the model to training mode
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.view(-1)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Get the class indices with maximum probability
            train_correct_predictions += torch.sum(predicted == labels.float())
            train_total_samples += labels.size(0)

        # Calculate training loss and accuracy for the epoch
        train_loss = running_loss / len(train_loader)
        train_accuracy = (train_correct_predictions / train_total_samples).item()

        # Validation
          
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0


        with torch.no_grad():
            for inputs, labels in val_loader:

                outputs = model(inputs)
                labels = labels.view(-1)

                val_loss = criterion(outputs, labels.long())
                val_running_loss += val_loss.item()

                _, val_predicted = torch.max(outputs.data, 1)
                val_correct_predictions += torch.sum(val_predicted == labels.float())
                val_total_samples += labels.size(0)

        # Calculate validation loss and accuracy for the epoch
        val_loss = val_running_loss / len(val_loader)
        val_accuracy = (val_correct_predictions / val_total_samples).item()

        # Print validation accuracy
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Check for improvement in validation accuracy
        if val_accuracy > best_val_accuracy:
            # Save the model with the best validation accuracy
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model)  # Create a deep copy of the model
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

def inference(test_data_input_coordinates, model):
    """
    Perform inference for testing data using the saved model.
    """
    model.eval()
    predictions_final_list =[]
    test_data_input_coordinates = test_data_input_coordinates.permute(0, 2, 1)
    # Forward pass
    with torch.no_grad():
        outputs = model(test_data_input_coordinates)
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

    # Preprocess input data
    input_tensor_combined = DataProcessor.process_input_data(training_data)

    # Preprocess coordinates data
    input_coordinate_tensor_combined = DataProcessor.process_coordinates_data(training_data, input_tensor_combined)

    # Normalize data
    normalized_input_coordinate_tensor = DataProcessor.normalize_input_coordinate_data(input_coordinate_tensor_combined)
   

    #Target labels
    input_target_per_country = DataProcessor.get_target_lables(training_data)
    
    # Define model parameters
    features = normalized_input_coordinate_tensor.shape[2]
    timesteps = normalized_input_coordinate_tensor.shape[1]
    num_classes = 2
    hidden_size = 16
    model = TempCNN(features, hidden_size, timesteps, num_classes)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_combined_tensor = torch.tensor(normalized_input_coordinate_tensor).float().to(device)
    Y_combined_tensor = torch.tensor(input_target_per_country).long().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
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

    # Preprocess testing data
    country_datasets = DataProcessor.testing_process_data(testing_data)

    test_normalized_input_coordinate_data = {}

    for country, _ in country_datasets.items():
        print(f'>>>>>>>>>>>>>>>>>>>>>>>{country}<<<<<<<<<<<<<<<<<<<<<<<<<')
        input_data = country_datasets[country]['input_data'].values
        input_data = torch.tensor(input_data, dtype=torch.float32)
        coordinate_value = country_datasets[country]
        input_coordinate_tensor_combined = DataProcessor.process_coordinates_data(coordinate_value, input_data)
        test_normalized_input_coordinate_data[country] = DataProcessor. normalize_input_coordinate_data(input_coordinate_tensor_combined)
        
    for i in range(5):
        # Train and evaluate the model
        print(f'--------------------------EXPERIMENT {i+1}:--------------------------------')
        num_epochs = 15
        patience = 5
        save_path = f'/netscratch/mudraje/spatial-explicit-ml/scripts/models/coordinates/IFcoordinates_best_model_exp_{i+1}.pt'  # Define save path for this experiment
        model = train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, save_path)
    
        print('-------------------------------------TRAINING COMPLETED--------------------------------------------')

  
        # Perform inference for each country
        for country, _ in country_datasets.items():
            predictions = inference(test_normalized_input_coordinate_data[country], model)
            target_country = country_datasets[country]['target'].values
            accuracy = perclass_accuracy(predictions, target_country)
            print(f'{country}: {accuracy}')

        print('---------------------------------------TESTING COMPLETED-----------------------------------')
    testing_data.close()
    training_data.close()