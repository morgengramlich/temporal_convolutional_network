import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad1 = nn.ConstantPad1d(((self.kernel_size-1)*dilation, 0), 0.0)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=dilation,
        ))
        self.dropout1 = nn.Dropout(p=0.2)
        self.pad2 = nn.ConstantPad1d(((self.kernel_size-1)*dilation, 0), 0.0)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.input_size,
            kernel_size=self.kernel_size,
            dilation=dilation,
        ))
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        output = self.pad1(x)
        output = F.relu(self.conv1(output))
        output = self.dropout1(output)
        output = self.pad2(output)
        output = F.relu(self.conv2(output))
        output = self.dropout2(output)
        return F.relu(x + output)

class TCN(nn.Module):
    def __init__(self, input_size, kernel_sizes, dilations):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.residuals = nn.Sequential()
        for n in range(len(kernel_sizes)):
            self.residuals.append(ResidualBlock(self.input_size, self.input_size, self.kernel_sizes[n], self.dilations[n]))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.input_size, self.input_size)
  
    def forward(self, value):
        output = torch.permute(value, (0, 2, 1))
        output = self.residuals(output)
        output = torch.permute(output, (0, 2, 1))
        output = self.dropout(output)
        output = self.fc(output)
        return output
