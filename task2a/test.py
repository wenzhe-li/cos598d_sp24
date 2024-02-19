import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def main(rank, world_size):
    # Initialize the distributed environment.
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:6585', rank=rank, world_size=world_size)

    dataset = MyDataset(list(range(1000)))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=100, sampler=sampler)

    for data in dataloader:
        # process data
        print(data)
        break

if __name__ == "__main__":
    world_size = 2
    for rank in range(world_size):
        # create processes for each rank
        p = Process(target=main, args=(rank, world_size))
        p.start()