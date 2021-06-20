import torch


class SUBatchSampler(torch.utils.data.Sampler):
  def __init__(
      self,
      cumulative_sizes,
      supervised_batch_size,
      unsupervised_batch_size,
      ):
    assert len(cumulative_sizes) ==2, 'Expected two supervised unsupervised dataset.'
    self.supervised_sampler = torch.utils.data.RandomSampler(range(cumulative_sizes[0]))
    self.unsupervised_sampler = torch.utils.data.SubsetRandomSampler(range(cumulative_sizes[0], cumulative_sizes[1]))

    self.isupervised_sampler = iter(self.supervised_sampler)
    self.iunsupervised_sampler = iter(self.unsupervised_sampler)

    self.supervised_batch_size = supervised_batch_size
    self.unsupervised_batch_size = unsupervised_batch_size

  def __len__(self):
    return (len(self.supervised_sampler) + self.supervised_batch_size - 1) // self.supervised_batch_size

  def __iter__(self):
    for _ in range(self.__len__()):
      batch = []

      for _ in range(self.supervised_batch_size):
        try:
          idx = next(self.isupervised_sampler)
        except:
          self.isupervised_sampler = iter(self.supervised_sampler)
          idx = next(self.isupervised_sampler)
        batch.append(idx)

      for _ in range(self.unsupervised_batch_size):
        try:
          idx = next(self.iunsupervised_sampler)
        except:
          self.iunsupervised_sampler = iter(self.unsupervised_sampler)
          idx = next(self.iunsupervised_sampler)
        batch.append(idx)

      yield batch
