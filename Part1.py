import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchinfo import summary 
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

# data loaders for CIFAR-10
train_loader_cifar = DataLoader(train_dataset_cifar, batch_size=64, shuffle=True, num_workers=2)
test_loader_cifar = DataLoader(test_dataset_cifar, batch_size=64, shuffle=False, num_workers=2)

# CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #architecture here
        
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
        #forward pass here
        
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

# Instantiate the CNN model and move it to the device
model_cifar = CNN().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_cifar = optim.SGD(model_cifar.parameters(), lr=0.001, momentum=0.9)

# Display the summary of the model
#summary(model_cifar, (3, 224, 224))


if __name__ == '__main__':
    # Check if CUDA (GPU) is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Display the summary of the model
    #summary(model_cifar, (3, 224, 224))

    # Training loop for CIFAR-10
    epochs = 7

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        total_batches = len(train_loader_cifar)

        # Iterate over batches in the training loader
        for i, data in enumerate(train_loader_cifar, 0):

            # Extract inputs (images) and labels from the current batch
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the gradients before backpropagation
            optimizer_cifar.zero_grad()

            # Forward pass: compute model predictions
            outputs = model_cifar(inputs)

            # Compute the loss between model predictions and ground truth labels
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients of the loss with respect to model parameters
            loss.backward()

            # Update the model parameters using the optimizer
            optimizer_cifar.step()

            # Update running loss for logging and visualization
            running_loss += loss.item()

            # Compute the number of correct predictions for the current batch
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


    # Testing phase

    # Set the model to evaluation mode (important for layers like dropout)
    model_cifar.eval()

    # Use torch.no_grad() to disable gradient computation during testing
    with torch.no_grad():
        # Initialize variables to track correct predictions and total samples during testing
        correct_predictions_test = 0
        total_samples_test = 0

        # Iterate over batches in the test loader
        for data_test in test_loader_cifar:
            # Extract inputs (images) and labels from the current test batch
            inputs_test, labels_test = data_test[0].to(device), data_test[1].to(device)

            # Forward pass: compute model predictions for the test inputs
            outputs_test = model_cifar(inputs_test)

            # Extract the predicted class labels by finding the indices of the maximum values in outputs
            _, predicted_test = torch.max(outputs_test.data, 1)

            # Update the total number of test samples processed so far
            total_samples_test += labels_test.size(0)

            # Update the number of correct predictions by comparing predicted and true labels
            correct_predictions_test += (predicted_test == labels_test).sum().item()

    # Calculate the testing accuracy by dividing the correct predictions by the total test samples
    test_accuracy = correct_predictions_test / total_samples_test

    # Print the testing accuracy
    print(f'Testing Accuracy: {test_accuracy:.3f}')

print('Finished Training and Testing on CIFAR-10')