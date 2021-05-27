import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Discriminator_Law(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(Discriminator_Law, self).__init__()
        self.problem = problem
        dim1 = 10
        dim2 = 16
        finaldim = 10
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 1)   # output layer
        self.dropout = nn.Dropout(0.75)

    def forward(self, x):
        x = self.hidden(x)
        # x = self.dropout(x)
        # x = F.leaky_relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.predict(x)
        return x



class Discriminator_Adult_Aw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(Discriminator_Adult_Aw, self).__init__()
        self.problem = problem
        dim1 = 128
        dim2 = 64
        finaldim = 64
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 2)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.sig = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)

        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)

        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)

        x = self.predict(x)
        x = self.soft(x)

        return x


class Discriminator_Adult_Ag(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(Discriminator_Adult_Ag, self).__init__()
        self.problem = problem
        dim1 = 128
        dim2 = 64
        finaldim = 64
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 2)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.sig = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = F.leaky_relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)
        for i in range(10):
            x = F.leaky_relu(self.hidden2(x))
            # x = self.dropout(x)
            x = self.batchnorm2(x)
        x = self.predict(x)
        # x = self.dropout(x)
        x = self.soft(x)
        # x = self.dropout(x)

        return x
