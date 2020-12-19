import torch


class StackedLSTM(torch.nn.Module):
    def __init__(self, input_dim):
        super(StackedLSTM, self).__init__()
        self.input_dim = input_dim
        self.rnn = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=3,
            bidirectional=True,
        )
        self.fc = torch.nn.Linear(256, 1)
        # self.dropout = torch.nn.Dropout(0.5)

    def forward(self, inputs): # inputs (64, 9, 5) (batch, seq_len - 1, input_dim)
        x = inputs.transpose(0, 1) # x (9, 64, 5) (seq_len -1, batch_size, input_dim)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x) # outs (9, 64, 32) (seq_len - 1, batch_size, hidden_size)
        output = self.fc(outs[-1]) # output (64, 1)
        output = output.view(-1)
        return output
