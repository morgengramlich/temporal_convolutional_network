import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    '''Residual block to use in TCN'''

    def __init__(self, input_size, hidden_size, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad1 = nn.ConstantPad1d(((self.kernel_size-1)*dilation, 0), 0.0)
        self.conv1 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=dilation,
        ))
        self.dropout1 = nn.Dropout1d(p=0.2)
        self.pad2 = nn.ConstantPad1d(((self.kernel_size-1)*dilation, 0), 0.0)
        self.conv2 = torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=dilation,
        ))
        self.dropout2 = nn.Dropout1d(p=0.2)
        self.identity_conv = None
        if self.input_size > 1 and self.input_size != self.hidden_size:
            self.identity_conv = nn.Conv1d(
                in_channels=self.input_size,
                out_channels=self.hidden_size,
                kernel_size=1,
            )

    def forward(self, x):
        '''One step of computation'''
        output = self.pad1(x)
        output = F.leaky_relu(self.conv1(output))
        output = self.dropout1(output)
        output = self.pad2(output)
        output = F.leaky_relu(self.conv2(output))
        output = self.dropout2(output)
        if self.input_size > 1 and self.input_size != self.hidden_size:
            x = self.identity_conv(x)
        return F.leaky_relu(x + output)

class TCN(nn.Module):
    '''Temporal Convolutional Network'''

    def __init__(self, input_size, num_filters, kernel_sizes, dilations):
        super(TCN, self).__init__()
        if len(num_filters) != len(kernel_sizes):
            raise ValueError('output_sizes and kernel_sizes must be of the same size')
        if len(kernel_sizes) != len(dilations):
            raise ValueError('kernel_sizes and dilations must be of the same size')
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.residuals = nn.Sequential()
        for n in range(len(kernel_sizes)):
            if n == 0:
                self.residuals.append(ResidualBlock(self.input_size, self.num_filters[n], self.kernel_sizes[n], self.dilations[n]))
            else:
                self.residuals.append(ResidualBlock(self.num_filters[n-1], self.num_filters[n], self.kernel_sizes[n], self.dilations[n]))
  
    def forward(self, value):
        '''One step of computation'''
        output = self.residuals(value)
        return output
