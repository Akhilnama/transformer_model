import torch

class TextDataset:

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def get_batch(self, batch_size):

        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])

        return x, y