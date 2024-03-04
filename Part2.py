import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn.functional as F

# Check if CUDA (GPU) is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resize images to 224x224 for training on CIFAR-10
transform_cifar = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download CIFAR-10 train and test datasets
train_dataset_cifar = CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
test_dataset_cifar = CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

# Create data loaders for CIFAR-10
train_loader_cifar = DataLoader(train_dataset_cifar, batch_size=32, shuffle=True, num_workers=1)
test_loader_cifar = DataLoader(test_dataset_cifar, batch_size=32, shuffle=False, num_workers=1)

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Your architecture here
        # Example:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)  # Adjusted the input size here
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        # Your forward pass here
        # Example:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.view(-1, 512 * 7 * 7)  # Adjusted the size here

        # Flatten the tensor before passing it through fully connected layers
        #x = x.view(x.size(0), -1)
         
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    def get_mid_layer_output(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Calculate the number of channels in the output of the previous layer
        num_channels_prev_layer = x.size(1)

        mid_layer_output = self.conv4(x)
        return mid_layer_output, num_channels_prev_layer
    
  

# Instantiate the CNN model and move it to the device
model_cifar = CNN().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_cifar = optim.SGD(model_cifar.parameters(), lr=0.001, momentum=0.9)

# Define the loss function and optimizer for MLP
#criterion_mlp = nn.CrossEntropyLoss()
#optimizer_mlp = optim.SGD(mlp_classifier.parameters(), lr=0.001, momentum=0.9)

# Display the summary of the model
#summary(model_cifar, (3, 224, 224))


if __name__ == '__main__':
    # Check if CUDA (GPU) is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Display the summary of the model
    #summary(model_cifar, (3, 224, 224))

    # Training loop for CIFAR-10
    epochs = 1
    accumulation_steps = 4  # Adjust this value based on your available GPU memory

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        #total_batches = len(train_loader_cifar)

        total_batches = 100  # For demonstration purposes, we will only train for 100 batches

        train_features = []
        train_labels = []

        for i, data in enumerate(train_loader_cifar, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            mid_layer_features, _ = model_cifar.get_mid_layer_output(inputs)

            train_features.append(mid_layer_features.cpu().detach().numpy())
            train_labels.append(labels.cpu().detach().numpy())
            torch.cuda.empty_cache()

            outputs = model_cifar(inputs)
            loss = criterion(outputs, labels)

            # Gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer_cifar.step()
                optimizer_cifar.zero_grad()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (i + 1) % 200 == 0:  # Print every 200 batches
                batch_accuracy = correct_predictions / total_samples
                print(f'Epoch {epoch + 1}, Batch {i + 1}/{total_batches}, Loss: {running_loss / 200:.3f}, Accuracy: {batch_accuracy:.3f}')
                running_loss = 0.0
                correct_predictions = 0
                total_samples = 0

        # Print overall training accuracy at the end of each epoch
        epoch_accuracy = correct_predictions / total_samples
        print(f'Epoch {epoch + 1} - Training Accuracy: {epoch_accuracy:.3f}')

        # Convert lists to numpy arrays
        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        # Create and train MLP classifier
        mlp_classifier = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=100)
        mlp_classifier.fit(train_features, train_labels)

    # Testing phase
    model_cifar.eval()
    with torch.no_grad():
        correct_predictions_test = 0
        total_samples_test = 0
        test_features = []
        test_labels = []

        for data_test in test_loader_cifar:
            inputs_test, labels_test = data_test[0].to(device), data_test[1].to(device)
            outputs_test = model_cifar(inputs_test)

            # Extract mid-layer features for testing
            mid_layer_features_test = model_cifar.get_mid_layer_output(inputs_test)
            test_features.append(mid_layer_features_test.cpu().detach().numpy())
            test_labels.append(labels_test.cpu().detach().numpy())

            _, predicted_test = torch.max(outputs_test.data, 1)
            total_samples_test += labels_test.size(0)
            correct_predictions_test += (predicted_test == labels_test).sum().item()

    # Convert lists to numpy arrays
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # Make predictions using the MLP classifier
    mlp_predictions = mlp_classifier.predict(test_features)

    # Calculate testing accuracy
    test_accuracy = accuracy_score(test_labels, mlp_predictions)
    print(f'MLP Testing Accuracy: {test_accuracy:.3f}')

    print('Finished Training and Testing on CIFAR-10')
