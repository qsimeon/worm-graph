import torch
import torch.nn as nn

class LinearNN(torch.nn.Module):
  def __init__(self, num_features):
    super(LinearNN, self).__init__()
    input_size = num_features
    output_size = input_size
    self.lin_model = nn.Linear(input_size, output_size)
  
  def forward(self, graph):
    x = graph.x
    x = self.lin_model(x)
    return x