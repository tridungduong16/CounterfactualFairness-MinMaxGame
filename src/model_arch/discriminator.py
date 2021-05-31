import torch 
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorLaw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorLaw, self).__init__()
        self.problem = problem
        dim1 = 16
        dim2 = 8
        finaldim = 16
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 1)   # output layer
        self.dropout = nn.Dropout(0.95)
        self.batchnorm = nn.BatchNorm1d(dim1)
        self.batchnorm1 = nn.BatchNorm1d(dim2)
        self.batchnorm2 = nn.BatchNorm1d(finaldim)
        self.laynorm = nn.LayerNorm(1)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        x = self.batchnorm(x)
        x = self.dropout(x)

        # x = F.leaky_relu(self.hidden1(x))
        # x = self.batchnorm1(x)
        # x = self.dropout(x)
        #
        # x = F.leaky_relu(self.hidden2(x))
        # x = self.batchnorm2(x)
        # x = self.dropout(x)

        x = self.predict(x)
        return x



class DiscriminatorAdultAw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorAdultAw, self).__init__()
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
        self.laynorm = nn.LayerNorm(64)

    def forward(self, x):
        """

        :param x:
        :return:
        """

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


class DiscriminatorAdultAg(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorAdultAg, self).__init__()
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
