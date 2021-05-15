import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



# class Generator(nn.Module):
#     def __init__(self, input_length: int):
#         super(Generator, self).__init__()
#         self.hidden = torch.nn.Linear(input_length, 128)   # hidden layer
#         self.hidden1 = torch.nn.Linear(1024, 512)   # hidden layer
#         self.hidden2 = torch.nn.Linear(512, 256)   # hidden layer
#         self.hidden3 = torch.nn.Linear(256, 128)   # hidden layer
#         self.hidden4 = torch.nn.Linear(128, 64)   # hidden layer
#         self.predict = torch.nn.Linear(64, 128)   # output layer
#         self.dropout = nn.Dropout(0.5)
#         self.activation = nn.Sigmoid()
#         self.mask = Variable(torch.tensor([1]*input_length), requires_grad=True)
#
#
#     def _independent_straight_through_sampling(self, rationale_logits):
#         """
#         Straight through sampling.
#         Outputs:
#             z -- shape (batch_size, sequence_length, 2)
#         """
#
#         z = torch.nn.functional.softmax(rationale_logits)
#         reduce_max = torch.max(z, dim=1, keepdim=False).values.reshape(-1, 1)
#         equal = z.eq(reduce_max)
#         z_hard = equal.type(torch.FloatTensor)
#         z_soft = (z_hard - z).detach_() + z
#
#         return z_soft
#
#
#     def forward(self, x):
#
#         x = self.x * self.mask
#         mask = self._independent_straight_through_sampling(x)
#         x = x*mask
#
#         x = self.hidden(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#
#         x = self.hidden4(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#
#         x = self.predict(x)
#         x = self.dropout(x)
#
#         return x
    

class Discriminator_Agnostic(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(Discriminator_Agnostic, self).__init__()
        self.problem = problem
        self.hidden = nn.Linear(input_length, 64)   # hidden layer
        self.hidden1 = nn.Linear(1024, 512)   # hidden layer
        self.hidden2 = nn.Linear(512, 256)   # hidden layer
        self.hidden3 = nn.Linear(256, 128)   # hidden layer
        self.hidden4 = nn.Linear(128, 64)   # hidden layer
        self.predict = nn.Linear(64, 1)   # output layer
        self.dropout = nn.Dropout(0.75)
        self.activation = nn.Sigmoid()
        self.bn1 = nn.BatchNorm2d(64), #applying batch norm
        self.ln1 = nn.LayerNorm(1)
        # self.bil = nn.Bilinear(input_length, 64)

        

    def forward(self, x):

        x = F.leaky_relu(self.hidden(x))
        x = self.dropout(x)
        x = self.predict(x)

        # x = self.bn1(F.relu(self.hidden(x)))

        # x = self.hidden(x)
        # print(x)
        # x = self.ln1(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        # x = self.activation(x)
        # x = self.hidden1(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        #
        # x = self.hidden2(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        #
        # x = self.hidden3(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        #
        #
        # x = self.hidden4(x)
        # x = self.dropout(x)
        # x = F.leaky_relu(x)

        # x = self.predict(x)
        # x = self.ln1(x)

        # x = self.dropout(x)
        # x = F.relu(x)
        return x
    

class Discriminator_Awareness(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(Discriminator_Awareness, self).__init__()
        self.problem = problem
        self.hidden = torch.nn.Linear(input_length, 64)   # hidden layer
        self.hidden1 = torch.nn.Linear(1024, 512)   # hidden layer
        self.hidden2 = torch.nn.Linear(512, 256)   # hidden layer
        self.hidden3 = torch.nn.Linear(256, 128)   # hidden layer
        self.hidden4 = torch.nn.Linear(128, 64)   # hidden layer
        self.predict = torch.nn.Linear(64, 1)   # output layer        
        self.dropout = nn.Dropout(0.95)
        self.activation = nn.Sigmoid()
        

    def forward(self, x):
        # x = torch.cat((Z,S),1)
        
        x = self.hidden(x)  
        x = self.dropout(x)
        x = F.relu(x)      

        # x = self.hidden1(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        #
        # x = self.hidden2(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        #
        # x = self.hidden3(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        #
        #
        # x = self.hidden4(x)
        # x = self.dropout(x)
        # x = F.relu(x)

        x = self.predict(x)             
        return x
    