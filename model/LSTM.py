import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hidden_dim = args.d_model
        self.num_layers = args.e_layers
        self.pred_len = args.pred_len
        self.lstm = nn.LSTM(args.factor, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, args.factor)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_dim).to(x_enc.device)
        c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_dim).to(x_enc.device)

        self.lstm.flatten_parameters()

        out, (hn, cn) = self.lstm(x_enc, (h0, c0))
        predictions = []
        input_seq = out[:, -1, :].unsqueeze(1)
        for _ in range(self.pred_len):
            pred = self.linear(input_seq)
            predictions.append(pred)
            input_seq, (hn, cn) = self.lstm(pred, (hn, cn))
        predictions = torch.cat(predictions, dim=1)

        return predictions