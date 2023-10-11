import torch
import torch.nn as nn
import torch.nn.functional as F


# Define various neural network architectures

class Net1(nn.Module):
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=1):
        super(Net1, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, 32)
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, 8)
        self.nonlin = nonlin
        self.output = nn.Linear(8, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = F.relu(self.dense2(X))
        X = F.relu(self.dense3(X))
        X = self.output(X)
        return X


    
class Net2(nn.Module):
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=1):
        super(Net2, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, 32)
        self.dense2 = nn.Linear(32, 16)
        self.nonlin = nonlin
        self.output = nn.Linear(16, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = F.relu(self.dense2(X))
        X = self.output(X)
        return X

class Net3(nn.Module):
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=1):
        super(Net3, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, 32)
        self.dense3 = nn.Linear(32, 8)
        self.nonlin = nonlin
        self.output = nn.Linear(8, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = F.relu(self.dense3(X))
        X = self.output(X)
        return X

class Net4(nn.Module):
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=1):
        super(Net4, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, 32)
        self.nonlin = nonlin
        self.output = nn.Linear(32, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X
    
class Net5(nn.Module):
    def __init__(self, input_size=12, num_units=64, nonlin=F.relu, nlabels=1):
        super(Net5, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, num_units)
        self.dense2 = nn.Linear(num_units, num_units)
        self.dense3 = nn.Linear(num_units, 32)
        self.dense4 = nn.Linear(32, 8)
        self.nonlin = nonlin
        self.output = nn.Linear(8, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        residual = X
        X = self.dense1(X)
        X = X +residual
        X = F.relu(X)
        residual = X
        X = self.dense2(X)
        X = X +residual
        X = F.relu(X)
        X = F.relu(self.dense3(X))
        X = F.relu(self.dense4(X))
        X = self.output(X)
        return X
    
class Net6(nn.Module):
    def __init__(self, input_size=12, num_units=128, nonlin=F.relu, nlabels=1):
        super(Net6, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, num_units)
        self.dense2 = nn.Linear(num_units, num_units)
        self.dense3 = nn.Linear(num_units, 32)
        self.dense4 = nn.Linear(32, 8)
        self.nonlin = nonlin
        self.output = nn.Linear(8, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        residual = X
        X = self.dense1(X)
        X = X +residual
        X = F.relu(X)
        residual = X
        X = self.dense2(X)
        X = X +residual
        X = F.relu(X)
        X = F.relu(self.dense3(X))
        X = F.relu(self.dense4(X))
        X = self.output(X)
        return X

class Net7(nn.Module):
    def __init__(self, input_size=12, num_units=16, nonlin=F.relu, nlabels=1):
        super(Net7, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.dense1 = nn.Linear(num_units, num_units)
        self.dense2 = nn.Linear(num_units, num_units)
        self.dense3 = nn.Linear(num_units, 32)
        self.dense4 = nn.Linear(32, 8)
        self.nonlin = nonlin
        self.output = nn.Linear(8, nlabels)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        residual = X
        X = self.dense1(X)
        X = X +residual
        X = F.relu(X)
        residual = X
        X = self.dense2(X)
        X = X +residual
        X = F.relu(X)
        X = F.relu(self.dense3(X))
        X = F.relu(self.dense4(X))
        X = self.output(X)
        return X

