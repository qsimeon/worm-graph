import torch
class BatchSampler(torch.utils.data.Sampler):
  def __init__(self, data_source):
    super(BatchSampler, self).__init__(data_source)
    self.data_source = data_source

  def __len__(self):
    return len(self.data_source)
  
  def __iter__(self):
    return iter(self.data_source)