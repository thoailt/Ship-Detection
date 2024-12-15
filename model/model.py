import torch
import torch.nn as nn

# Define the 3-layer CNN model
class CNN3Layer(nn.Module):
    def __init__(self):
        super(CNN3Layer, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: RGB, Output: 32 filters
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (32, 40, 40)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64 filters
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (64, 20, 20)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 128 filters
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (128, 10, 10)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 10 * 10, 256)  # Flattened input: 128*10*10, Output: 256 neurons
        self.fc2 = nn.Linear(256, 2)  # Binary classification (2 output neurons)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flatten the features for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Final output (no activation, handled by loss function)
        return x