class Dataset:

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors), "All tensors must have the same size in the first dimension."
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)