import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bias=True)
        self.fc = nn.Linear(self.seq_length*self.hidden_size, num_classes)

    def forward(self, x, state=None):
        '''Propagate input through LSTM'''
        lstm_out, state = self.lstm(x, state) # shape (N, L, H_out)
        lstm_out = lstm_out.contiguous().view(-1, self.seq_length*self.hidden_size)
        fc_out = self.fc(lstm_out) # powerful decoder
        return fc_out, state