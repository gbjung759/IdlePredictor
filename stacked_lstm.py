import torch


class StackedLSTM(torch.nn.Module):
    def __init__(self, input_dim):
        super(StackedLSTM, self).__init__()
        self.input_dim = input_dim
        self.rnn = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=128,
            num_layers=1,
            bidirectional=False,
            dropout=0.5
        )
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(128, 1)

    def forward(self, inputs):
        # inputs shape (batch, seq) (64, 50)
        x = inputs.transpose(0, 1)  # (batch, seq, embedding) -> (seq, batch, embedding) (50, 64, 128)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x) # outs (seq, batch, hidden * 2) (50, 64, 256)
        output = self.fc(outs[-1]) # (batch, outdim) (64, 8)
        return output
