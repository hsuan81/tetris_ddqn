import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions, in_channels=4, screen_shape=(84, 84)):
        super(DQN, self).__init__()
        in_channels = in_channels
        num_actions = num_actions
        screen_shape = screen_shape
        h = screen_shape[0]
        w = screen_shape[1]
        
        
        # could add batchnorm2d layers after each covnet if data volume is too large
        # see: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        
        # convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w, 3, 1), 4, 2), 3, 1)
        # convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h, 3, 1), 4, 2), 3, 1)

        convw = self.conv2d_size_out(self.conv2d_size_out(w, 3, 1), 4, 2)
        convh = self.conv2d_size_out(self.conv2d_size_out(h, 3, 1), 4, 2)

        self.fc1 = nn.Linear(convw * convh * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

        # self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(128)

        # convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w, 3, 1), 1, 1), 3, 1)
        # convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h, 3, 1), 1, 1), 3, 1)
        
        # self.fc1 = nn.Linear(convw * convh * 128, 512)
        # self.fc2 = nn.Linear(512, num_actions)
        
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # reshape the tensor to one dimension for fc layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)
    
    def conv2d_size_out(self, size, kernel_size, stride, padding_size=0):
        return (size + 2 * padding_size - (kernel_size - 1) - 1) // stride  + 1
    