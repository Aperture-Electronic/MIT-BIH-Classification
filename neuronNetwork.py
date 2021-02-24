import torch;
import torch.nn as nn;
import torch.nn.functional as nnfunc;

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        # Definition of batch normalization layers
        self.bn1 = nn.BatchNorm1d(1);
        self.bn2 = nn.BatchNorm1d(128);

        # Definition of convolution layers
        self.conv1 = nn.Conv1d(1, 16, 7); # Input 1x, Output 16x, Kernel 7
        self.conv2 = nn.Conv1d(16, 64, 3); # Input 16x, Output 64x, Kernel 3
        self.conv3 = nn.Conv1d(64, 128, 3); # Input 64x, Output 128x, Kernel 3

        # Definition of full-connected layers
        self.fc1 = nn.Linear(128 * 29, 400);
        self.fc2 = nn.Linear(400, 8);

    def forward(self, x):
        # Neuron network forward
        x = self.bn1(x);
        x = nnfunc.max_pool1d(nnfunc.relu(self.conv1(x)), 2);
        x = nnfunc.max_pool1d(nnfunc.relu(self.conv2(x)), 2);
        x = nnfunc.max_pool1d(nnfunc.relu(self.conv3(x)), 2);
        x = self.bn2(x);
        x = torch.flatten(x, 1);
        x = nnfunc.relu(self.fc1(x));
        x = nnfunc.softmax(self.fc2(x), dim = 1);

        return x;


