import torch
from torch import nn
from torch.nn.functional import relu

torch.manual_seed(0)


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        out, hidden = self.rnn(x, h0)
        # out = out[:,-1,:]
        out = self.fc(out)
        return out, hidden


class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)
        out, hidden = self.gru(x, h0)
        # out = out[:,-1,:]
        out = self.fc(out)
        return out, hidden


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)
        out, hidden = self.lstm(x, (h0, c0))
        # out = out[:,-1,:]
        out = self.fc(out)
        return out, hidden




class Model1(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=16, hidden2=16, hidden3=16):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_dim)
        self.out = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inputs):
        x = relu(self.fc1(inputs))
        # x = self.dropout(x)
        x = relu(self.fc2(x))
        # x = self.dropout(x)
        x = relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        return self.out(x)


Model2 = lambda x: nn.Sequential(nn.Linear(x[0], 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 500),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(500, 500),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(500, x[1]),
                                 nn.LogSoftmax(dim=1)
                                 )

Model3 = lambda x: nn.Sequential(
    nn.Linear(x[0], 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, x[1]),
    nn.LogSoftmax(dim=1)
)


class Model4(nn.Module):

    def __init__(self, input_dim, output_dim, hidden=64):
        super(Model4, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, hidden)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out
