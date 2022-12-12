import torch

class LinearNN(torch.nn.Module):
  def __init__(self, input_size):
    '''
    A simple linear regression model to use as a baseline.
    The output will be the same shape as the input.
    '''
    super(LinearNN, self).__init__()
    self.input_size = input_size
    self.output_size  = input_size
    self.linear = torch.nn.Linear(self.input_size, self.output_size)
  
  def loss_fn(self):
    '''The loss function to be used with this model.'''
    return torch.nn.MSELoss()

  def forward(self, input, tau=1):
    '''
    input: batch of data
    '''
    # Repeat for tau>0 offset target
    for i in range(tau):
      output = self.linear(input)
    return output