import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierNN(nn.Module):
    def __init__(self,
                 kernel_size=2,
                 init_channels=1,
                 image_channels=1,
                 output_dim=10,
                 learning_rate=0.0001):
        super(ClassifierNN, self).__init__()

        self.image_channels = image_channels  # Store image_channels as an attribute

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=image_channels,
                               out_channels=init_channels,
                               kernel_size=kernel_size,
                               stride=2)
        self.conv2 = nn.Conv2d(in_channels=init_channels,
                               out_channels=init_channels * 2,
                               kernel_size=kernel_size,
                               stride=2,
                               padding=0)
        self.conv3 = nn.Conv2d(in_channels=init_channels * 2,
                               out_channels=init_channels * 4,
                               kernel_size=kernel_size,
                               stride=2,
                               padding=0)
        self.conv4 = nn.Conv2d(in_channels=init_channels * 4,
                               out_channels=init_channels * 8,
                               kernel_size=kernel_size,
                               stride=2,
                               padding=0)

        # Calculate the output size after convolutions
        self.conv_output_size = self._get_conv_output_size()

        # Define fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def _get_conv_output_size(self):
        # Dummy input to calculate the output size after convolutions
        x = torch.randn(1, self.image_channels, 32, 32)  # Use self.image_channels here
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        conv_output_size = x.view(x.size(0), -1).size(1)
        return conv_output_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output