import math

from attention.attention import Attention
from src import USE_CUDA

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class PointerNet(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 use_cuda=USE_CUDA):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(seq_len, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, use_cuda=use_cuda)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, target):
        """
        Args:
            inputs: [batch_size x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        assert seq_len == self.seq_len

        embedded = self.embedding(inputs)
        target_embedded = self.embedding(target)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        loss = 0

        ret = []

        for i in range(seq_len):

            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)

            _, logits = self.pointer(query, encoder_outputs)
            logits = F.softmax(logits)

            ret.append(torch.argmax(logits).item())

            decoder_input = target_embedded[:, i, :]

            loss += self.criterion(logits, target[:, i])
        return loss / seq_len, ret