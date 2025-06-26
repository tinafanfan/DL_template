import torch
import torch.nn as nn

class FeedForwardNeuralNet(nn.Module):

    def __init__(self, args):
        super(FeedForwardNeuralNet, self).__init__()
        # define first layer
        self.l1 = nn.Linear(args.input_size, args.hidden_size)
        # activation function
        self.relu = nn.ReLU()
        # define second layer
        self.l2 = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, x):
        # x = x.to(torch.float32)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out
