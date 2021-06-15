import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class DiscriminatorLawAw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorLawAw, self).__init__()
        self.problem = problem
        dim1 = 32
        dim2 = 8
        finaldim = 32
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

        # x = F.leaky_relu(self.hidden2(x))
        # x = self.batchnorm2(x)
        # x = self.dropout(x)

        x = self.predict(x)
        return x

class DiscriminatorLaw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorLaw, self).__init__()
        self.problem = problem
        dim1 = 32
        dim2 = 16
        finaldim = 16
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 1)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(dim1)
        self.batchnorm1 = nn.BatchNorm1d(dim2)
        self.batchnorm2 = nn.BatchNorm1d(finaldim)
        self.laynorm = nn.LayerNorm(1)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        x = self.batchnorm(x)
        x = self.dropout(x)

        x = F.leaky_relu(self.hidden1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x)

        x = F.leaky_relu(self.hidden2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)

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

class DiscriminatorCompasAw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorCompasAw, self).__init__()
        self.problem = problem
        dim1 = 32
        dim2 = 32
        finaldim = 32
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 2)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.sig = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(dim1)
        self.batchnorm2 = nn.BatchNorm1d(dim2)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)

        # x = F.relu(self.hidden1(x))
        # x = self.dropout(x)
        # x = self.batchnorm2(x)

        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)

        x = self.predict(x)
        x = self.soft(x)

        return x

class DiscriminatorCompasAg(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorCompasAg, self).__init__()
        self.problem = problem
        dim1 = 32
        dim2 = 32
        finaldim = 32
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 2)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.sig = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(dim1)
        self.batchnorm2 = nn.BatchNorm1d(dim2)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)
        # x = F.leaky_relu(self.hidden1(x))
        # x = self.dropout(x)
        # x = self.batchnorm2(x)
        for i in range(2):
            x = F.leaky_relu(self.hidden2(x))
            x = self.dropout(x)
            x = self.batchnorm2(x)
        x = self.predict(x)
        x = self.soft(x)

        return x



def train_law(train_x, train_y):
    Net = DiscriminatorLaw(train_x.shape[1])
    data_set = TensorDataset(train_x, train_y)
    train_batches = DataLoader(data_set, batch_size=1024, shuffle=False)

    epochs = 500
    learning_rate = 1e-8

    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
    for i in range(epochs):
        for x_batch, y_batch in train_batches:
            optimizer.zero_grad()
            loss = loss_fn(Net(x_batch), y_batch)
            loss.backward()
            optimizer.step()
    return Net



